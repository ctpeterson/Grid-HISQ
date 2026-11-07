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
 * rational function of -D^2, with the light, strange and regulator masses entering as
 * constant shifts, and every field shares MILC's bounds on -D^2 with its own MD (force)
 * and GR/FA (heatbath, action) orders.
 */

#include <Grid/Grid.h>
#include <fstream>
#include <deque>
#include <limits>
#include <map>
#include <regex>
#include <sstream>

using namespace Grid;

////////////////////////////////////////////////////////////////
// Typedefs
////////////////////////////////////////////////////////////////
typedef GenericHMCRunner<InterleavedIntegrator> HMCWrapper;
typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;

typedef ImprovedStaggeredFermionD StaggeredFermionOperator;
typedef StaggeredImplD FermionImplPolicy;
typedef StaggeredEvenEvenRational<FermionImplPolicy> FermionAction;

typedef PeriodicGimplD Gimpl;

typedef XmlReader Serialiser;

////////////////////////////////////////////////////////////////
// MILC's combined light/strange/regulator pseudofermion
////////////////////////////////////////////////////////////////

// det(Q_f)^(Flavors_f/4) by bare mass
struct DeterminantFactor { RealD Mass; int Flavors; };

MultiShiftFunction combinedRationalFunction(
  const std::vector<DeterminantFactor>& factors,
  const StaggeredRationalActionParams& params,
  int invPow,
  bool action,
  bool inverse
) {
  const int degree = action ? params.action_degree : params.md_degree;
  const RealD tolerance = action ? params.action_tolerance : params.md_tolerance;

  std::vector<MultiShiftFunction> parts;
  RealD norm = 1.0;
  
  for (auto& factor : factors) {
    const RealD shift = 4.0 * factor.Mass * factor.Mass;
    const int power = inverse ? -factor.Flavors : factor.Flavors;
    
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
    
    parts.emplace_back(remez, tolerance, power < 0);
    norm *= parts.back().norm;
    for (auto& pole : parts.back().poles) { pole += shift; }
  }

  std::vector<std::pair<RealD, RealD>> terms; // (pole, residue)
  for (int f = 0; f < parts.size(); ++f) {
    for (int k = 0; k < parts[f].poles.size(); ++k) {
      const RealD pole = parts[f].poles[k];
      RealD residue = parts[f].residues[k];
      for (int g = 0; g < parts.size(); ++g) 
      { if (g != f) { residue *= parts[g].approx(-pole); } }
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

// One pseudofermion: the Naik epsilon of its operator and its three rational functions of
// -D^2, generated from the Remez settings in Params (XML) or read from MILC's rationals file
struct FieldSpec {
  RealD NaikEpsilon;
  StaggeredRationalActionParams Params;
  MultiShiftFunction HeatBath;
  MultiShiftFunction Action;
  MultiShiftFunction Deriv;
  bool ActionIsNorm = false;
};

struct Deck {
  HMCparameters HMC;
  CheckpointerParameters Checkpoint;
  bool ILDGCheckpoint = false;        // else Nersc
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
  deck.HMC.MD.MDsteps = HamiltonianMonteCarlo.MolecularDynamicsSteps;
  deck.HMC.MD.trajL = HamiltonianMonteCarlo.TrajectoryLength;
  deck.HMC.NoMetropolisUntil = HamiltonianMonteCarlo.NoMetropolisUntil;
  deck.HMC.StartingType = Initialization.StartType;
  deck.Checkpoint.config_prefix = Checkpoint.ConfigurationPrefix;
  deck.Checkpoint.rng_prefix = Checkpoint.RandomNumberGeneratorPrefix;
  deck.Checkpoint.saveInterval = Checkpoint.SaveInterval;
  deck.Checkpoint.format = Checkpoint.Format;
  deck.RNG.serial_seeds = RandomNumberGenerator.SerialSeeds;
  deck.RNG.parallel_seeds = RandomNumberGenerator.ParallelSeeds;
  deck.FermionStepMultiplicity = HamiltonianMonteCarlo.FermionActionStepMultiplicity;
  deck.GaugeStepMultiplicity = HamiltonianMonteCarlo.GaugeActionStepMultiplicity;
  deck.BareGaugeCoupling = Action.BareGaugeCoupling;
  deck.TadpoleFactor = Action.TadpoleFactor;

  // MILC's rationals file: bounds [lambda_low, lambda_high] on -D^2 for every pseudofermion,
  // order_MD for the force and order_GR = order_FA for heatbath and action.
  auto field = [&](
    RealD epsilon,
    const std::vector<DeterminantFactor>& factors,
    int actionDegree,
    int forceDegree
  ) {
    const StaggeredRationalActionParams params(
      factors[0].Flavors,
      RationalParams.LambdaLow,
      RationalParams.LambdaHigh,
      CGParams.MaxIterations,
      CGParams.ActionStoppingCondition,
      actionDegree,
      CGParams.ForceStoppingCondition,
      forceDegree,
      RationalParams.Precision,
      0
    );
    return FieldSpec{
      epsilon, params,
      combinedRationalFunction(factors, params, 8, true, false),
      combinedRationalFunction(factors, params, 4, true, true),
      combinedRationalFunction(factors, params, 4, false, true)
    };
  };

  // Five pseudofermions as in MILC's in.m013m065m838 / rationals.sample.su3_rhmc_hisq_f211:
  //   1    det(Q_l)^(2/4) det(Q_s)^(1/4) det(Q_r)^(-3/4), one rational function of -D^2
  //   2-4  det(Q_r)^(1/4) each, cancelling the regulator denominator
  //   5    det(Q_c)^(1/4) on the Naik-epsilon links (Eq. A6 uses the bare charm mass)
  const RealD ml = Action.LightFermionMass;
  const RealD ms = Action.StrangeFermionMass;
  const RealD mr = Action.RegulatorFermionMass;
  const RealD mc = Action.CharmFermionMass;
  const FieldSpec lightStrange = field(
    0.0, {{ml, 2}, {ms, 1}, {mr, -3}},
    RationalParams.LightStrangeActionDegree, RationalParams.LightStrangeForceDegree
  );
  const FieldSpec regulator = field(
    0.0, {{mr, 1}}, RationalParams.RegulatorActionDegree, RationalParams.RegulatorForceDegree
  );
  const FieldSpec charm = field(
    calcNaikEpsilon(mc), {{mc, 1}},
    RationalParams.CharmActionDegree, RationalParams.CharmForceDegree
  );
  deck.Fields = {lightStrange, regulator, regulator, regulator, charm};

  // MILC's total_dyn_flavors: the sea quarks l, s, c; the regulator determinants cancel.
  deck.SeaFlavors = 2 + 1 + 1;
  return deck;
}

// MILC's ks_imp_rhmc input deck (setup.c, readin) and rationals file (load_rhmc_params.c):
// "key value(s)" in a fixed order, "#" to the end of the line is a comment
class MILCReader {
public:
  MILCReader(const std::string& file) {
    std::cout << GridLogMessage << "Reading MILC input " << file << std::endl;
    std::ifstream in(file);
    GRID_ASSERT(in.is_open() && "cannot open MILC input");
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream ss(line.substr(0, line.find('#')));
      std::string token;
      while (ss >> token) { _tokens.push_back(token); }
    }
  }

  std::string peek() const { return _pos < _tokens.size() ? _tokens[_pos] : ""; }

  std::string word() {
    GRID_ASSERT(_pos < _tokens.size() && "unexpected end of MILC input");
    return _tokens[_pos++];
  }

  void expect(const std::string& key) {
    const std::string got = word();
    if (got != key) {
      std::cout << GridLogError << "expected " << key << ", got " << got << std::endl;
      GRID_ASSERT(0);
    }
  }

  int integer(const std::string& key) { expect(key); return std::stoi(word()); }
  RealD real(const std::string& key) { expect(key); return std::stod(word()); }

  std::vector<RealD> reals(const std::string& key, int n) {
    std::vector<RealD> values;
    expect(key);
    for (int i = 0; i < n; ++i) { values.push_back(std::stod(word())); }
    return values;
  }

private:
  std::vector<std::string> _tokens;
  size_t _pos = 0;
};

// One function of the rationals file, prod_k (x + 4 m_k^2)^(y_k/z_k) as
// res[0] + sum_k res[k]/(x + pole[k]) with pole[0] a dummy, in MILC's x = -D^2 on even sites
MultiShiftFunction readMILCFunction(MILCReader& in, const std::string& tag, RealD tolerance) {
  in.reals("y_" + tag, 4);
  in.reals("z_" + tag, 4);
  in.reals("m_" + tag, 4);
  const int order = in.integer("order_" + tag);
  MultiShiftFunction f(order, 0.0, 0.0);
  f.order = order;
  f.norm = in.real("res_" + tag);
  for (auto& residue : f.residues) { residue = in.real("res_" + tag); }
  in.real("pole_" + tag);
  for (auto& pole : f.poles) { pole = in.real("pole_" + tag); }
  f.tolerances.assign(order, tolerance);
  return f;
}

Deck readMILCDeck(const std::string& file) {
  MILCReader in(file);
  Deck deck;

  in.integer("prompt");
  const Coordinate lattice(
    {in.integer("nx"), in.integer("ny"), in.integer("nz"), in.integer("nt")}
  );
  GRID_ASSERT(lattice == GridDefaultLatt() && "--grid must match nx ny nz nt");
  if (in.peek() == "node_geometry") { in.reals("node_geometry", 4); }
  if (in.peek() == "ionode_geometry") { in.reals("ionode_geometry", 4); }
  const int seed = in.integer("iseed");
  deck.RNG.parallel_seeds = std::to_string(seed);   // MILC rng: iseed, serial {iseed, volume}
  deck.RNG.serial_seeds = std::to_string(seed) + " " + std::to_string(
    lattice[0] * lattice[1] * lattice[2] * lattice[3]
  );
  const int nPseudo = in.integer("n_pseudo");
  in.expect("load_rhmc_params");
  std::string rationals = in.word();
  deck.BareGaugeCoupling = in.real("beta");
  const int nDyn = in.integer("n_dyn_masses");
  in.reals("dyn_mass", nDyn);
  deck.SeaFlavors = 0;
  for (RealD flavors : in.reals("dyn_flavors", nDyn)) { deck.SeaFlavors += flavors; }
  deck.TadpoleFactor = in.real("u0");
  deck.HMC.Trajectories = in.integer("warms");
  deck.HMC.Trajectories += in.integer("trajecs");
  deck.HMC.NoMetropolisUntil = 0;                 // MILC's warms are Metropolis trajectories
  in.integer("traj_between_meas");                  // Grid measures every trajectory
  const RealD timeStep = in.real("microcanonical_time_step");
  const int milcSteps = in.integer("steps_per_trajectory");
  GRID_ASSERT(milcSteps % 2 == 0 && "MILC 3G1F requires an even number of steps");
  deck.HMC.MD.MDsteps = milcSteps / 2;              // complete Omelyan periods
  deck.HMC.MD.trajL = timeStep * milcSteps;
  deck.FermionStepMultiplicity = 1;                 // MILC's INT_3G1F
  deck.GaugeStepMultiplicity = 3;

  // the rationals file, relative to the working directory as in MILC, else to the deck
  if (!std::ifstream(rationals) && file.rfind('/') != std::string::npos) {
    rationals = file.substr(0, file.rfind('/') + 1) + rationals;
  }
  MILCReader rat(rationals);
  GRID_ASSERT(rat.integer("n_pseudo") == nPseudo && "n_pseudo differs from rationals file");
  for (int i = 0; i < nPseudo; ++i) {
    // md, fa, gr: relative residuals
    const std::vector<RealD> residual = in.reals("cgresid_md_fa_gr", 3);
    const std::vector<RealD> iterations = in.reals("max_multicg_md_fa_gr", 3);
    in.reals("cgprec_md_fa_gr", 3);                 // double precision throughout

    FieldSpec field;
    field.NaikEpsilon = rat.real("naik_term_epsilon");
    // R = prod_f Q_f^(-Flavors_f/4): MD force; FA applies R^(1/2).
    field.Deriv = readMILCFunction(rat, "MD", residual[0]);
    field.HeatBath = readMILCFunction(rat, "GR", residual[2]); // R^(-1/2): heatbath
    field.Action = readMILCFunction(rat, "FA", residual[1]);
    field.ActionIsNorm = true;
    field.Params = StaggeredRationalActionParams(
      0,                                              // nf, lo, hi: logging only
      0.0,
      0.0,
      std::max({iterations[0], iterations[1], iterations[2]}),
      residual[1],
      field.Action.order,
      residual[0],
      field.Deriv.order,
      0,
      0
    );
    deck.Fields.push_back(field);
  }

  in.integer("prec_ff");
  const int nPbp = in.integer("number_of_pbp_masses"); // measurement code, not run here
  if (nPbp > 0) {
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
  auto ildg = [](const std::string& name) {
    return name.find("ildg") != std::string::npos || name.find("scidac") != std::string::npos;
  };

  const std::regex trajectory("\\.([0-9]+)$");
  std::string prefix;
  const std::string start = in.word();
  if (start == "fresh") { deck.HMC.StartingType = "ColdStart"; }
  else if (start == "warm") { deck.HMC.StartingType = "TepidStart"; }
  else if (start.rfind("reload_", 0) == 0) {
    const std::string name = in.word();
    std::smatch match;
    GRID_ASSERT(std::regex_search(name, match, trajectory) &&
                "reload needs <prefix>.<trajectory>");
    prefix = match.prefix();
    deck.HMC.StartingType = "CheckpointStart";
    deck.HMC.StartTrajectory = std::stoi(match[1]);
    deck.ILDGCheckpoint = ildg(name);
  } else { GRID_ASSERT(0 && "starting lattice command not supported"); }

  const std::string save = in.word();
  if (save == "forget") { deck.Checkpoint.saveInterval = std::numeric_limits<int>::max(); }
  else if (save.rfind("save_", 0) == 0) {
    // MILC continues as reload lat.100, save lat.110: Grid appends the trajectory itself
    prefix = std::regex_replace(in.word(), trajectory, "");
    deck.ILDGCheckpoint = ildg(save);
  } else { GRID_ASSERT(0 && "saving lattice command not supported"); }

  deck.Checkpoint.config_prefix = prefix;
  deck.Checkpoint.rng_prefix = prefix + ".rng";
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
  deck.HMC.MD.name = "InterleavedIntegrator";
  HMCWrapper TheHMC(deck.HMC);

  TheHMC.ReadCommandLine(argc, argv);
  TheHMC.Resources.AddFourDimGrid("gauge");
  if (deck.ILDGCheckpoint) { TheHMC.Resources.LoadILDGCheckpointer(deck.Checkpoint); }
  else { TheHMC.Resources.LoadNerscCheckpointer(deck.Checkpoint); }
  TheHMC.Resources.SetRNGSeeds(deck.RNG);

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();

  ////////////////////////////////////////////////////////////////
  // Fermion operators: one per Naik epsilon
  ////////////////////////////////////////////////////////////////
  std::vector<Complex> boundary = {1, 1, 1, -1};
  StaggeredFermionOperator::ImplParams Params(boundary);
  HISQConfiguration<HMCWrapper::ImplPolicy> Policy(GridPtr);

  // Match MILC: M = 2m + Dslash, so double Grid's hopping coefficients. This also
  // matches MILC pseudofermion normalization. As in MILC the operators are massless:
  // the rational functions are of -D^2, with each bare mass in their poles as (2m)^2.
  std::map<RealD, StaggeredFermionOperator> Operators;

  ////////////////////////////////////////////////////////////////
  // Full action
  ////////////////////////////////////////////////////////////////
  ActionLevel<HMCWrapper::Field> Level1(deck.FermionStepMultiplicity);
  ActionLevel<HMCWrapper::Field> Level2(deck.GaugeStepMultiplicity);

  std::deque<FermionAction> Fields;
  for (const auto& spec : deck.Fields) {
    auto& Op = Operators.try_emplace(
      spec.NaikEpsilon,
      *GridPtr, *GridRBPtr, 0.0, 2.0, -(1.0 + spec.NaikEpsilon) / 12.0, 1.0, Params
    ).first->second;
    Fields.emplace_back(Op, spec.Params, spec.HeatBath, spec.Action, spec.Deriv, spec.ActionIsNorm);
    Fields.back().signContract(Policy.promise(spec.NaikEpsilon));
    Level1.push_back(&Fields.back());
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
  TheHMC.Run(Policy, [](auto& md) {
    for (int level = 0; level < md.levels; ++level)
      md.schemes[level] = omelyan2(md.T, md.V[level], md.as[level].multiplier, 0.2);
    md.evolution.set_branches({&md.schemes[0], &md.schemes[1]});
  });

  Grid_finalize();
}

/* MILC input (pass as --milc <deck>, with --grid equal to nx ny nz nt): the ks_imp_rhmc deck
   of su3_rhmc_hisq (ks_imp_rhmc/test/su3_rhmc_hisq.f211.1.sample-in) and the rationals file it
   names in load_rhmc_params. Each pseudofermion uses the massless operator with the
   naik_term_epsilon of the file; GR is the heatbath, MD the force, and FA the
   half-power used to evaluate the action as |FA Phi|^2; cgresid_md_fa_gr are the
   relative residuals of the multishift solves, max_multicg_md_fa_gr their iteration limits;
   warms + trajecs trajectories with the Metropolis test throughout; MILC's
   steps_per_trajectory/2 complete Omelyan periods of 2*microcanonical_time_step,
   lambda = 0.2, and independent fermion/gauge multiplicities 1 and 3. XML counts
   complete periods directly; TrajectoryLength/MolecularDynamicsSteps is their length;
   iseed seeds the MILC rng (serial {iseed, volume}); dyn_flavors give the one-loop gauge
   coefficients. fresh is ColdStart, warm TepidStart; reload_* needs a lattice named
   <prefix>.<trajectory> that Grid's checkpointer can read (the ILDG checkpointer for names
   containing ildg or scidac, MILC's native format cannot be read); save_* writes Grid's
   checkpoints every trajectory under the given name, less any .<trajectory> ending, forget
   writes nothing. One checkpointer both reads and writes, so a continuation keeps its prefix
   (reload_serial lat.100, save_serial lat.110). traj_between_meas, cgprec_md_fa_gr,
   prec_ff and the pbp block are read and ignored; continue is not supported.
*/

/* Example XML input file (pass as --xml <xml-file-name>.xml):
   The masses, beta, and u0 are the a ~ 0.12 fm physical-mass ensemble in
   Table IV of Phys. Rev. D 87, 054505. The regulator mass is 0.2 as in
   Phys. Rev. D 82, 074501. Step counts are examples, not a reproduction of a MILC
   production input deck; check them before use. The rational bounds, the regulator and
   charm orders and the rng seeds follow in.m013m065m838 (1e-15, 90, orders 7/9). The
   combined field approximates each of its three factors separately at the given degree
   (3 x degree poles); its light factor x^(2/4) on [4 m_l^2, 90] reaches 1e-9 at degree 20
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
