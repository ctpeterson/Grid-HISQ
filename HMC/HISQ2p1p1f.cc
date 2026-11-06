/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./HMC/HISQ2p1p1f.cc

Copyright (C) 2015-2016

Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */

/**
 * @file HISQ2p1p1f.cc
 * @author Curtis Taylor Peterson
 * @brief RHMC for 2+1+1 HISQ fermions
 * @details The action is specified in Appendix A of
 * "Scaling studies of QCD with the dynamical highly improved staggered quark action",
 * https://doi.org/10.1103/PhysRevD.82.074501.
 * The five-field Hasenbusch factorization follows Sec. II of that paper
 * and the physical-mass ensembles in https://doi.org/10.1103/PhysRevD.87.054505:
 * one combined light/strange ratio, three one-flavor regulators, and charm.
 * The pseudofermions follow MILC's ks_imp_rhmc rationals file (e.g.
 * ks_imp_rhmc/test/rationals.sample.su3_rhmc_hisq_f211): the combined field is one
 * rational function of the light operator, with the strange and regulator operators
 * entering through their masses as constant shifts, and every field shares MILC's
 * bounds on -D^2 with its own MD (force) and GR/FA (heatbath, action) orders.
 */

#include <Grid/Grid.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

using namespace Grid;

////////////////////////////////////////////////////////////////
// Typedefs
////////////////////////////////////////////////////////////////
typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;

typedef ImprovedStaggeredFermionD StaggeredFermionOperator;
typedef StaggeredImplD FermionImplPolicy;
typedef StaggeredEvenEvenRational<FermionImplPolicy> FermionAction;

template<class Impl> struct DeterminantFactor;
typedef DeterminantFactor<FermionImplPolicy> Factor;

typedef PeriodicGimplD Gimpl;

typedef XmlReader Serialiser;

////////////////////////////////////////////////////////////////
// MILC's combined light/strange/regulator pseudofermion
////////////////////////////////////////////////////////////////

template<class Impl>
struct DeterminantFactor { FermionOperator<Impl>& Op; int Flavors; };

template<class Impl>
MultiShiftFunction combinedRationalFunction(
  const std::vector<DeterminantFactor<Impl>>& factors,
  const StaggeredRationalActionParams& params,
  int invPow,
  bool action,
  bool inverse
) {
  const int degree = action ? params.action_degree : params.md_degree;
  const RealD tolerance = action ? params.action_tolerance : params.md_tolerance;
  const RealD m2 = factors[0].Op.Mass() * factors[0].Op.Mass();

  std::vector<RealD> shifts;
  std::vector<MultiShiftFunction> parts;
  
  for (auto& factor : factors) {
    const RealD shift = factor.Op.Mass() * factor.Op.Mass() - m2;
    const int power = inverse ? -factor.Flavors : factor.Flavors;
    
    GRID_ASSERT(params.lo + shift > 0.0 && "Remez interval must lie above every mass");
    
    std::cout << GridLogMessage
              << "Generating degree " << degree << " approximation for (x+" << shift
              << ")^(" << power << "/" << invPow << ")" << std::endl;
    
    AlgRemez remez(params.lo + shift, params.hi + shift, params.precision);
    
    double error = remez.generateApprox(degree, std::abs(factor.Flavors), invPow);
    if (error > tolerance) {
      std::cout << GridLogMessage
                << "WARNING: Remez approximation has a larger error "
                << error << " than the CG tolerance " << tolerance
                << "! Try increasing the number of poles" << std::endl;
    }
    
    MultiShiftFunction part;
    part.Init(remez, tolerance, power < 0);
    shifts.push_back(shift);
    parts.push_back(part);
  }

  std::vector<std::pair<RealD, RealD>> terms; // (pole, residue)
  RealD norm = 1.0;
  for (auto& part : parts) { norm *= part.norm; }
  for (int f = 0; f < parts.size(); ++f) {
    for (int k = 0; k < parts[f].poles.size(); ++k) {
      const RealD pole = parts[f].poles[k] + shifts[f];
      RealD residue = parts[f].residues[k];
      for (int g = 0; g < parts.size(); ++g) 
      { if (g != f) { residue *= parts[g].approx(shifts[g] - pole); } }
      terms.push_back({pole, residue});
  } }
  std::sort(terms.begin(), terms.end()); // smallest pole first for ConjugateGradientMultiShift

  MultiShiftFunction combined(terms.size(), params.lo, params.hi);
  combined.order = terms.size();
  combined.norm = norm;
  for (int i = 0; i < terms.size(); ++i) {
    combined.poles[i] = terms[i].first;
    combined.residues[i] = terms[i].second;
    combined.tolerances[i] = tolerance;
  }
  return combined;
}

////////////////////////////////////////////////////////////////
// Serializable structs (for reading inputs)
////////////////////////////////////////////////////////////////

#define GRID_SERIALIZABLE(StructName, ...)                                        \
  struct StructName: Serializable {                                               \
    GRID_SERIALIZABLE_CLASS_MEMBERS(StructName, __VA_ARGS__);                     \
    template <class ReaderClass>                                                  \
    StructName(Reader<ReaderClass>& Reader) { read(Reader, #StructName, *this); } \
  };

GRID_SERIALIZABLE(InitializationParameters, std::string, StartType);

GRID_SERIALIZABLE(
  CheckpointParameters,
  std::string, Format,
  std::string, ConfigurationPrefix,
  std::string, RandomNumberGeneratorPrefix,
  int, SaveInterval
);

GRID_SERIALIZABLE(
  RandomNumberGeneratorParameters,
  std::string, SerialSeeds,
  std::string, ParallelSeeds
);

GRID_SERIALIZABLE(
  HamiltonianMonteCarloParameters,
  int, NoMetropolisUntil,
  
  double, TrajectoryLength,
  int, MolecularDynamicsSteps,
  
  int, FermionActionStepMultiplicity,
  int, GaugeActionStepMultiplicity
);

GRID_SERIALIZABLE(
  ActionParameters,
  double, BareGaugeCoupling,
  double, TadpoleFactor,
  double, LightFermionMass,
  double, StrangeFermionMass,
  double, CharmFermionMass,
  double, RegulatorFermionMass
);

GRID_SERIALIZABLE(
  ConjugateGradientParameters,
  double, ActionStoppingCondition,
  double, ForceStoppingCondition,
  int, MaxIterations
);

// MILC rationals file: lambda_low/lambda_high bound -D^2 for every pseudofermion;
// each pseudofermion has its own order_MD (force) and order_GR = order_FA (heatbath, action)
GRID_SERIALIZABLE(
  RationalApproximationParameters,
  double, LambdaLow,
  double, LambdaHigh,
  int, Precision,
  int, LightStrangeActionDegree,
  int, LightStrangeForceDegree,
  int, RegulatorActionDegree,
  int, RegulatorForceDegree,
  int, CharmActionDegree,
  int, CharmForceDegree
);

#undef GRID_SERIALIZABLE

////////////////////////////////////////////////////////////////
// Input deck: the XML file, or MILC's input deck and rationals file
////////////////////////////////////////////////////////////////

// One pseudofermion: the bare mass and Naik epsilon of its operator, its determinant factors
// prod_f det(Q_f)^(Flavors_f/4) by bare mass, and its three rational functions in the variable
// of the operator's MdagM, generated from the Remez settings in Params (XML) or read from
// MILC's rationals file (HasFunctions)
struct FieldSpec {
  RealD Mass;
  RealD NaikEpsilon;
  std::vector<std::pair<RealD, int>> Factors;
  StaggeredRationalActionParams Params;
  bool HasFunctions = false;
  MultiShiftFunction HeatBath;
  MultiShiftFunction Action;
  MultiShiftFunction Deriv;
};

struct Deck {
  Coordinate Lattice;                 // MILC deck only, checked against --grid
  HMCparameters HMC;
  CheckpointerParameters Checkpoint;
  std::string CheckpointFormat;       // Nersc or ILDG
  RNGModuleParameters RNG;
  int FermionStepMultiplicity;
  int GaugeStepMultiplicity;
  RealD BareGaugeCoupling;
  RealD TadpoleFactor;
  int SeaFlavors;                     // MILC's total_dyn_flavors in the one-loop coefficients
  std::vector<FieldSpec> Fields;
};

Deck readXMLDeck(const std::string& file) {
  Serialiser Reader(file, false, "grid");

  InitializationParameters Initialization(Reader);
  HamiltonianMonteCarloParameters HamiltonianMonteCarlo(Reader);
  ActionParameters Action(Reader);
  CheckpointParameters Checkpoint(Reader);
  RandomNumberGeneratorParameters RandomNumberGenerator(Reader);
  ConjugateGradientParameters CGParams(Reader);
  RationalApproximationParameters RationalParams(Reader);

  Deck deck;
  deck.HMC.MD.name = std::string("MinimumNorm2");
  deck.HMC.MD.MDsteps = HamiltonianMonteCarlo.MolecularDynamicsSteps;
  deck.HMC.MD.trajL = HamiltonianMonteCarlo.TrajectoryLength;
  deck.HMC.NoMetropolisUntil = HamiltonianMonteCarlo.NoMetropolisUntil;
  deck.HMC.StartingType = Initialization.StartType;
  deck.Checkpoint.config_prefix = Checkpoint.ConfigurationPrefix;
  deck.Checkpoint.rng_prefix = Checkpoint.RandomNumberGeneratorPrefix;
  deck.Checkpoint.saveInterval = Checkpoint.SaveInterval;
  deck.Checkpoint.format = Checkpoint.Format;
  deck.CheckpointFormat = "Nersc";
  deck.RNG.serial_seeds = RandomNumberGenerator.SerialSeeds;
  deck.RNG.parallel_seeds = RandomNumberGenerator.ParallelSeeds;
  deck.FermionStepMultiplicity = HamiltonianMonteCarlo.FermionActionStepMultiplicity;
  deck.GaugeStepMultiplicity = HamiltonianMonteCarlo.GaugeActionStepMultiplicity;
  deck.BareGaugeCoupling = Action.BareGaugeCoupling;
  deck.TadpoleFactor = Action.TadpoleFactor;

  // MILC's rationals file: bounds [lambda_low, lambda_high] on -D^2 for every pseudofermion,
  // order_MD for the force and order_GR = order_FA for heatbath and action. For the operator
  // with (doubled) mass 2m, MdagM = -D^2 + (2m)^2 on the even grid, so the bounds move by (2m)^2.
  auto rationalParams = [&](RealD mass, int flavors, int actionDegree, int forceDegree) {
    const RealD m2 = 4.0 * mass * mass;
    return StaggeredRationalActionParams(
      flavors,
      RationalParams.LambdaLow + m2,
      RationalParams.LambdaHigh + m2,
      CGParams.MaxIterations,
      CGParams.ActionStoppingCondition,
      actionDegree,
      CGParams.ForceStoppingCondition,
      forceDegree,
      RationalParams.Precision,
      0
    );
  };

  // Five pseudofermions as in MILC's in.m013m065m838 / rationals.sample.su3_rhmc_hisq_f211:
  //   1    det(Q_l)^(2/4) det(Q_s)^(1/4) det(Q_r)^(-3/4), one rational function of Q_l
  //   2-4  det(Q_r)^(1/4) each, cancelling the regulator denominator
  //   5    det(Q_c)^(1/4) on the Naik-epsilon links (Eq. A6 uses the bare charm mass)
  const RealD ml = Action.LightFermionMass;
  const RealD ms = Action.StrangeFermionMass;
  const RealD mr = Action.RegulatorFermionMass;
  const RealD mc = Action.CharmFermionMass;
  FieldSpec lightStrange = {
    ml, 0.0, {{ml, 2}, {ms, 1}, {mr, -3}},
    rationalParams(ml, 2, RationalParams.LightStrangeActionDegree, RationalParams.LightStrangeForceDegree)
  };
  FieldSpec regulator = {
    mr, 0.0, {{mr, 1}},
    rationalParams(mr, 1, RationalParams.RegulatorActionDegree, RationalParams.RegulatorForceDegree)
  };
  FieldSpec charm = {
    mc, calcNaikEpsilon(mc), {{mc, 1}},
    rationalParams(mc, 1, RationalParams.CharmActionDegree, RationalParams.CharmForceDegree)
  };
  deck.Fields = {lightStrange, regulator, regulator, regulator, charm};

  // MILC's total_dyn_flavors: the sea quarks l, s, c; the regulator determinants cancel.
  deck.SeaFlavors = 2 + 1 + 1;
  return deck;
}

// MILC's ks_imp_rhmc input deck (setup.c, readin) and rationals file (load_rhmc_params.c):
// "key value(s)" in a fixed order, "#" to the end of the line is a comment
class MILCReader {
public:
  MILCReader(const std::string& file): _file(file) {
    std::ifstream in(file);
    if (!in) {
      std::cout << GridLogError << "cannot open MILC input " << file << std::endl;
      GRID_ASSERT(0);
    }
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream ss(line.substr(0, line.find('#')));
      std::string token;
      while (ss >> token) { _tokens.push_back(token); }
    }
  }

  bool done() const { return _pos >= _tokens.size(); }
  std::string peek() const { return done() ? std::string() : _tokens[_pos]; }

  std::string word() {
    if (done()) {
      std::cout << GridLogError << _file << ": unexpected end of input" << std::endl;
      GRID_ASSERT(0);
    }
    return _tokens[_pos++];
  }

  void expect(const std::string& key) {
    const std::string got = word();
    if (got != key) {
      std::cout << GridLogError << _file << ": expected " << key << ", got " << got << std::endl;
      GRID_ASSERT(0);
    }
  }

  int integer(const std::string& key) { expect(key); return std::stoi(word()); }
  RealD real(const std::string& key) { expect(key); return std::stod(word()); }
  std::string string(const std::string& key) { expect(key); return word(); }

  std::vector<int> integers(const std::string& key, int n) {
    std::vector<int> values;
    expect(key);
    for (int i = 0; i < n; ++i) { values.push_back(std::stoi(word())); }
    return values;
  }

  std::vector<RealD> reals(const std::string& key, int n) {
    std::vector<RealD> values;
    expect(key);
    for (int i = 0; i < n; ++i) { values.push_back(std::stod(word())); }
    return values;
  }

private:
  std::string _file;
  std::vector<std::string> _tokens;
  size_t _pos = 0;
};

// One function of the rationals file: prod_k (x + 4 m_k^2)^(y_k/z_k) as res[0] + sum_k res[k]/(x + pole[k]),
// pole[0] a dummy, in MILC's x = -D^2 on even sites
struct MILCRational {
  std::vector<int> y;
  std::vector<int> z;
  std::vector<RealD> m;
  int order;
  std::vector<RealD> res;
  std::vector<RealD> pole;
};

MILCRational readMILCRational(MILCReader& in, const std::string& tag) {
  MILCRational f;
  f.y = in.integers("y_" + tag, 4);
  f.z = in.integers("z_" + tag, 4);
  f.m = in.reals("m_" + tag, 4);
  f.order = in.integer("order_" + tag);
  f.res.resize(f.order + 1);
  f.pole.resize(f.order + 1);
  for (int k = 0; k <= f.order; ++k) { f.res[k] = in.real("res_" + tag); }
  for (int k = 0; k <= f.order; ++k) { f.pole[k] = in.real("pole_" + tag); }
  return f;
}

// the operator with (doubled) mass 2m has MdagM = x + (2m)^2, so the poles move by -(2m)^2
MultiShiftFunction shiftedFunction(const MILCRational& f, RealD shift, RealD tolerance) {
  MultiShiftFunction g(f.order, shift, shift);
  g.order = f.order;
  g.norm = f.res[0];
  for (int k = 0; k < f.order; ++k) {
    g.poles[k] = f.pole[k + 1] - shift;
    g.residues[k] = f.res[k + 1];
    g.tolerances[k] = tolerance;
  }
  return g;
}

Deck readMILCDeck(const std::string& file) {
  MILCReader in(file);
  Deck deck;

  in.integer("prompt");
  deck.Lattice = Coordinate({in.integer("nx"), in.integer("ny"), in.integer("nz"), in.integer("nt")});
  if (in.peek() == "node_geometry") { in.integers("node_geometry", 4); }
  if (in.peek() == "ionode_geometry") { in.integers("ionode_geometry", 4); }
  const int seed = in.integer("iseed");
  const int nPseudo = in.integer("n_pseudo");
  std::string rationals = in.string("load_rhmc_params");
  deck.BareGaugeCoupling = in.real("beta");
  const int nDyn = in.integer("n_dyn_masses");
  in.reals("dyn_mass", nDyn);
  const std::vector<int> dynFlavors = in.integers("dyn_flavors", nDyn);
  deck.TadpoleFactor = in.real("u0");
  const int warms = in.integer("warms");
  const int trajecs = in.integer("trajecs");
  in.integer("traj_between_meas");                  // Grid measures every trajectory
  const RealD timeStep = in.real("microcanonical_time_step");
  const int steps = in.integer("steps_per_trajectory");
  std::vector<std::vector<RealD>> residual(nPseudo); // md, fa, gr: relative residuals
  std::vector<std::vector<int>> iterations(nPseudo);
  for (int i = 0; i < nPseudo; ++i) {
    residual[i] = in.reals("cgresid_md_fa_gr", 3);
    iterations[i] = in.integers("max_multicg_md_fa_gr", 3);
    in.integers("cgprec_md_fa_gr", 3);              // double precision throughout
  }
  in.integer("prec_ff");
  const int nPbp = in.integer("number_of_pbp_masses"); // measurement code, not run here
  if (nPbp > 0 || in.peek() == "max_cg_prop") {
    in.integer("max_cg_prop");
    in.integer("max_cg_prop_restarts");
    in.integer("npbp_reps");
    in.integer("prec_pbp");
  }
  for (int i = 0; i < nPbp; ++i) {
    in.real("mass");
    in.real("naik_term_epsilon");
    in.real("error_for_propagator");
    in.real("rel_error_for_propagator");
  }

  // Grid's checkpointer names lattices <prefix>.<trajectory>; a reloaded lattice must follow
  // that, Grid cannot read MILC's native format (ILDG and SciDAC files are read by name)
  auto split = [](const std::string& name, std::string& prefix, int& trajectory) {
    const size_t dot = name.rfind('.');
    if (dot == std::string::npos) return false;
    const std::string tail = name.substr(dot + 1);
    if (tail.empty() || tail.find_first_not_of("0123456789") != std::string::npos) return false;
    prefix = name.substr(0, dot);
    trajectory = std::stoi(tail);
    return true;
  };
  auto format = [](const std::string& name) -> std::string {
    if (name.find("ildg") != std::string::npos || name.find("scidac") != std::string::npos) return "ILDG";
    return "Nersc";
  };

  const int total = warms + trajecs;
  deck.HMC.StartTrajectory = 0;
  deck.HMC.Trajectories = total;
  deck.HMC.NoMetropolisUntil = 0;                 // MILC's warms are Metropolis trajectories
  deck.CheckpointFormat = "Nersc";
  deck.Checkpoint.saveInterval = std::numeric_limits<int>::max(); // forget: never saves
  const std::string start = in.word();
  if (start == "fresh") { deck.HMC.StartingType = "ColdStart"; }
  else if (start == "warm") { deck.HMC.StartingType = "TepidStart"; }
  else if (start.rfind("reload_", 0) == 0) {
    const std::string name = in.word();
    int trajectory;
    if (!split(name, deck.Checkpoint.config_prefix, trajectory)) {
      std::cout << GridLogError << "reload: rename " << name << " to <prefix>.<trajectory>" << std::endl;
      GRID_ASSERT(0);
    }
    deck.HMC.StartingType = "CheckpointStart";
    deck.HMC.StartTrajectory = trajectory;
    deck.CheckpointFormat = format(name);
    deck.Checkpoint.rng_prefix = deck.Checkpoint.config_prefix + ".rng";
  } else {
    std::cout << GridLogError << "starting lattice command " << start << " is not supported" << std::endl;
    GRID_ASSERT(0);
  }
  const std::string save = in.word();
  if (save.rfind("save_", 0) == 0) {
    const std::string name = in.word();
    if (save.find("ildg") != std::string::npos) { in.word(); } // ILDG LFN
    deck.Checkpoint.config_prefix = name;
    deck.Checkpoint.rng_prefix = name + ".rng";
    deck.Checkpoint.saveInterval = 1;               // MILC saves once at the end
    deck.CheckpointFormat = format(save);
  } else if (save != "forget") {
    std::cout << GridLogError << "saving lattice command " << save << " is not supported" << std::endl;
    GRID_ASSERT(0);
  }

  deck.HMC.MD.name = std::string("MinimumNorm2");
  deck.HMC.MD.MDsteps = steps;
  deck.HMC.MD.trajL = timeStep * steps;
  deck.FermionStepMultiplicity = 1;                 // MILC's INT_3G1F
  deck.GaugeStepMultiplicity = 3;
  deck.RNG.parallel_seeds = std::to_string(seed);   // MILC rng: iseed, serial {iseed, volume}
  deck.RNG.serial_seeds = std::to_string(seed) + " " + std::to_string(
    deck.Lattice[0] * deck.Lattice[1] * deck.Lattice[2] * deck.Lattice[3]
  );
  deck.SeaFlavors = 0;
  for (int flavors : dynFlavors) { deck.SeaFlavors += flavors; }

  // the rationals file, relative to the working directory as in MILC, else to the deck
  if (!std::ifstream(rationals) && file.rfind('/') != std::string::npos) {
    rationals = file.substr(0, file.rfind('/') + 1) + rationals;
  }
  MILCReader rat(rationals);
  GRID_ASSERT(rat.integer("n_pseudo") == nPseudo && "n_pseudo differs between deck and rationals");
  for (int i = 0; i < nPseudo; ++i) {
    FieldSpec field;
    field.NaikEpsilon = rat.real("naik_term_epsilon");
    const MILCRational md = readMILCRational(rat, "MD"); // R = prod_f Q_f^(-Flavors_f/4): force and action
    const MILCRational gr = readMILCRational(rat, "GR"); // R^(-1/2): heatbath
    const MILCRational fa = readMILCRational(rat, "FA"); // R^(1/2): MILC's |FA Phi|^2, unused here
    for (int k = 0; k < 4; ++k) {
      if (md.y[k] != 0) { field.Factors.push_back({md.m[k], -md.y[k] * 4 / md.z[k]}); }
    }
    GRID_ASSERT(!field.Factors.empty() && "pseudofermion without a mass in the rationals file");
    field.Mass = field.Factors[0].first;
    const RealD shift = 4.0 * field.Mass * field.Mass;
    field.HeatBath = shiftedFunction(gr, shift, residual[i][2]);
    field.Action = shiftedFunction(md, shift, residual[i][1]);
    field.Deriv = shiftedFunction(md, shift, residual[i][0]);
    field.HasFunctions = true;
    field.Params = StaggeredRationalActionParams(
      field.Factors[0].second,
      shift,
      shift + 90.0,                                   // MILC's usual lambda_high, logging only
      std::max({iterations[i][0], iterations[i][1], iterations[i][2]}),
      residual[i][1],
      gr.order,
      residual[i][0],
      md.order,
      0,
      0
    );
    std::cout << GridLogMessage << "MILC pseudofermion " << i + 1 << ": naik_term_epsilon "
              << field.NaikEpsilon << ", orders MD " << md.order << " GR " << gr.order
              << " FA " << fa.order << " (FA unused)" << std::endl;
    deck.Fields.push_back(field);
  }
  return deck;
}

////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  ////////////////////////////////////////////////////////////////
  // Input: --xml <file>, or --milc <deck> with MILC's input deck and rationals file
  ////////////////////////////////////////////////////////////////
  const bool milc = GridCmdOptionExists(argv, argv + argc, "--milc");
  Deck deck = milc ? readMILCDeck(GridCmdOptionPayload(argv, argv + argc, "--milc"))
                   : readXMLDeck(GridCmdOptionPayload(argv, argv + argc, "--xml"));

  /////////////////////////////////////////////////////////////////
  // Hamiltonian Monte Carlo wrapper
  /////////////////////////////////////////////////////////////////
  HMCWrapper TheHMC(deck.HMC);

  TheHMC.ReadCommandLine(argc, argv);
  TheHMC.Resources.AddFourDimGrid("gauge");
  if (deck.CheckpointFormat == "ILDG") { TheHMC.Resources.LoadILDGCheckpointer(deck.Checkpoint); }
  else { TheHMC.Resources.LoadNerscCheckpointer(deck.Checkpoint); }
  TheHMC.Resources.SetRNGSeeds(deck.RNG);

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();
  for (int mu = 0; mu < deck.Lattice.size(); ++mu) {
    GRID_ASSERT(GridPtr->FullDimensions()[mu] == deck.Lattice[mu] && "--grid must match nx ny nz nt");
  }

  ////////////////////////////////////////////////////////////////
  // Fermion operators: one per (bare mass, Naik epsilon), smeared links per Naik epsilon
  ////////////////////////////////////////////////////////////////
  std::vector<Complex> boundary = {1, 1, 1, -1};
  StaggeredFermionOperator::ImplParams Params(boundary);
  HISQConfiguration<HMCWrapper::ImplPolicy> Policy(GridPtr);

  // Match MILC: M = 2m + Dslash, so double Grid's mass and hopping
  // coefficients. This also matches MILC pseudofermion normalization.
  // Deck masses remain bare am; MILC's rational bounds are on -D^2 = MdagM - (2m)^2.
  std::map<std::pair<RealD, RealD>, std::unique_ptr<StaggeredFermionOperator>> Operators;
  std::map<RealD, Primals<HMCWrapper::Field>> Links;
  auto OperatorFor = [&](RealD mass, RealD epsilon) -> StaggeredFermionOperator& {
    auto& op = Operators[{mass, epsilon}];
    if (!op) {
      op.reset(new StaggeredFermionOperator(
        *GridPtr, *GridRBPtr, 2.0 * mass, 2.0, -(1.0 + epsilon) / 12.0, 1.0, Params
      ));
    }
    return *op;
  };

  ////////////////////////////////////////////////////////////////
  // Full action
  ////////////////////////////////////////////////////////////////
  ActionLevel<HMCWrapper::Field> Level1(deck.FermionStepMultiplicity);
  ActionLevel<HMCWrapper::Field> Level2(deck.GaugeStepMultiplicity);

  std::vector<std::unique_ptr<FermionAction>> Fields;
  for (auto& spec : deck.Fields) {
    StaggeredFermionOperator& Op = OperatorFor(spec.Mass, spec.NaikEpsilon);
    if (!spec.HasFunctions) {
      std::vector<Factor> factors;
      for (auto& f : spec.Factors) { factors.push_back({OperatorFor(f.first, spec.NaikEpsilon), f.second}); }
      spec.HeatBath = combinedRationalFunction(factors, spec.Params, 8, true, false);
      spec.Action = combinedRationalFunction(factors, spec.Params, 4, true, true);
      spec.Deriv = combinedRationalFunction(factors, spec.Params, 4, false, true);
    }
    if (!Links.count(spec.NaikEpsilon)) { Links[spec.NaikEpsilon] = Policy.promise(spec.NaikEpsilon); }
    Fields.emplace_back(new FermionAction(Op, spec.Params, spec.HeatBath, spec.Action, spec.Deriv));
    Fields.back()->signContract(Links[spec.NaikEpsilon]);
    Level1.push_back(Fields.back().get());
  }

  // Eq. A2, beta = 10/g^2. Tadpole improvement is gauge-only.
  const RealD u0 = deck.TadpoleFactor;
  const int nf = deck.SeaFlavors;
  PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl> GaugeAction(
    GridPtr, 
    deck.BareGaugeCoupling, 
    1.0, 
    oneLoopMILCRectangleCoefficient(u0, nf), 
    oneLoopMILCParallelogramCoefficient(u0, nf)
  );
  Level2.push_back(&GaugeAction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);

  //////////////////////////////////////////////////////////////
  // Run HMC with HISQ smearing
  //////////////////////////////////////////////////////////////
  TheHMC.Resources.AddObservable<PlaqObs>();
  TheHMC.Run(Policy);

  Grid_finalize();
}

/* MILC input (pass as --milc <deck>, with --grid equal to nx ny nz nt): the ks_imp_rhmc deck
   of su3_rhmc_hisq (ks_imp_rhmc/test/su3_rhmc_hisq.f211.1.sample-in) and the rationals file it
   names in load_rhmc_params. Each pseudofermion's operator is the first mass of its y_MD list,
   with the naik_term_epsilon of the file; GR is the heatbath, MD the force and the action
   (FA, MILC's half-power for |FA Phi|^2, is read and ignored); cgresid_md_fa_gr are the
   relative residuals of the multishift solves, max_multicg_md_fa_gr their iteration limits;
   warms + trajecs trajectories with the Metropolis test throughout; steps_per_trajectory
   MinimumNorm2 steps of microcanonical_time_step with MILC's INT_3G1F multiplicities 1 and 3;
   iseed seeds the MILC rng (serial {iseed, volume}); dyn_flavors give the one-loop gauge
   coefficients. fresh is ColdStart, warm TepidStart; reload_* needs a lattice named
   <prefix>.<trajectory> that Grid's checkpointer can read (the ILDG checkpointer for names
   containing ildg or scidac, MILC's native format cannot be read); save_* writes Grid's checkpoints every
   trajectory under the given name, forget writes nothing. traj_between_meas, cgprec_md_fa_gr,
   prec_ff and the pbp block are read and ignored; continue is not supported.
*/

/* Example XML input file (pass as --xml <xml-file-name>.xml):
   The masses, beta, and u0 are the a ~ 0.12 fm physical-mass ensemble in
   Table IV of Phys. Rev. D 87, 054505. The regulator mass is 0.2 as in
   Phys. Rev. D 82, 074501. Step counts are examples, not a reproduction of a MILC
   production input deck; check them before use. The rational bounds, the regulator and
   charm orders and the rng seeds follow in.m013m065m838 (1e-15, 90, orders 7/9). The
   combined field approximates each of its three factors separately at the given degree
   (3 x degree poles); its light factor x^(2/4) on [4 m_l^2, 90] reaches 5e-9 at degree 20
   with 80 digits. With --enable-rng=milc the parallel rng takes one seed and the serial rng
   {seed, volume} (4096 below is 8^4).
<?xml version="1.0"?>
<grid>
  <InitializationParameters>
    <StartType>TepidStart</StartType>
  </InitializationParameters>
  <HamiltonianMonteCarloParameters>
    <NoMetropolisUntil>0</NoMetropolisUntil>
    <TrajectoryLength>1.0</TrajectoryLength>
    <MolecularDynamicsSteps>15</MolecularDynamicsSteps>
    <FermionActionStepMultiplicity>1</FermionActionStepMultiplicity>
    <GaugeActionStepMultiplicity>3</GaugeActionStepMultiplicity>
  </HamiltonianMonteCarloParameters>
  <ActionParameters>
    <BareGaugeCoupling>6.0</BareGaugeCoupling>
    <TadpoleFactor>0.86372</TadpoleFactor>
    <LightFermionMass>0.00184</LightFermionMass>
    <StrangeFermionMass>0.0507</StrangeFermionMass>
    <CharmFermionMass>0.628</CharmFermionMass>
    <RegulatorFermionMass>0.2</RegulatorFermionMass>
  </ActionParameters>
  <CheckpointParameters>
    <Format>IEEE64BIG</Format>
    <ConfigurationPrefix>ckpoint_hmc_lat</ConfigurationPrefix>
    <RandomNumberGeneratorPrefix>ckpoint_hmc_rng</RandomNumberGeneratorPrefix>
    <SaveInterval>1</SaveInterval>
  </CheckpointParameters>
  <RandomNumberGeneratorParameters>
    <SerialSeeds>6 4096</SerialSeeds>
    <ParallelSeeds>6</ParallelSeeds>
  </RandomNumberGeneratorParameters>
  <ConjugateGradientParameters>
    <ActionStoppingCondition>1e-10</ActionStoppingCondition>
    <ForceStoppingCondition>1e-8</ForceStoppingCondition>
    <MaxIterations>10000</MaxIterations>
  </ConjugateGradientParameters>
  <RationalApproximationParameters>
    <LambdaLow>1e-15</LambdaLow>
    <LambdaHigh>90.0</LambdaHigh>
    <Precision>80</Precision>
    <LightStrangeActionDegree>20</LightStrangeActionDegree>
    <LightStrangeForceDegree>16</LightStrangeForceDegree>
    <RegulatorActionDegree>9</RegulatorActionDegree>
    <RegulatorForceDegree>7</RegulatorForceDegree>
    <CharmActionDegree>9</CharmActionDegree>
    <CharmForceDegree>7</CharmForceDegree>
  </RationalApproximationParameters>
</grid>
*/
