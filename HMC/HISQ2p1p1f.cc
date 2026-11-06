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
// Main
////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  
  ////////////////////////////////////////////////////////////////
  // Hamiltonian Monte Carlo setup
  ////////////////////////////////////////////////////////////////
  Serialiser Reader(GridCmdOptionPayload(argv, argv + argc, "--xml"), false, "grid");

  InitializationParameters Initialization(Reader);
  HamiltonianMonteCarloParameters HamiltonianMonteCarlo(Reader);
  ActionParameters Action(Reader);
  CheckpointParameters Checkpoint(Reader);
  RandomNumberGeneratorParameters RandomNumberGenerator(Reader);
  ConjugateGradientParameters CGParams(Reader);
  RationalApproximationParameters RationalParams(Reader);

  IntegratorParameters MDParams;
  HMCparameters HMCParams;
  CheckpointerParameters CPParams;
  RNGModuleParameters RNGParams;

  MDParams.name = std::string("MinimumNorm2");
  MDParams.MDsteps = HamiltonianMonteCarlo.MolecularDynamicsSteps;
  MDParams.trajL = HamiltonianMonteCarlo.TrajectoryLength;

  HMCParams.NoMetropolisUntil = HamiltonianMonteCarlo.NoMetropolisUntil;
  HMCParams.StartingType = Initialization.StartType;
  HMCParams.MD = MDParams;

  CPParams.config_prefix = Checkpoint.ConfigurationPrefix;
  CPParams.rng_prefix = Checkpoint.RandomNumberGeneratorPrefix;
  CPParams.saveInterval = Checkpoint.SaveInterval;
  CPParams.format = Checkpoint.Format;

  RNGParams.serial_seeds = RandomNumberGenerator.SerialSeeds;
  RNGParams.parallel_seeds = RandomNumberGenerator.ParallelSeeds;

  /////////////////////////////////////////////////////////////////
  // Hamiltonian Monte Carlo wrapper
  /////////////////////////////////////////////////////////////////
  HMCWrapper TheHMC(HMCParams);

  TheHMC.ReadCommandLine(argc, argv);
  TheHMC.Resources.AddFourDimGrid("gauge");
  TheHMC.Resources.LoadNerscCheckpointer(CPParams);
  TheHMC.Resources.SetRNGSeeds(RNGParams);

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();

  ////////////////////////////////////////////////////////////////
  // Fermion operators
  ////////////////////////////////////////////////////////////////
  std::vector<Complex> boundary = {1, 1, 1, -1};
  StaggeredFermionOperator::ImplParams Params(boundary);

  // Match MILC: M = 2m + Dslash, so double Grid's mass and hopping
  // coefficients. This also matches MILC pseudofermion normalization.
  // XML masses remain bare am; MILC's rational bounds are on -D^2 = MdagM - (2m)^2.
  // Eq. A6 uses the bare charm mass, not the doubled constructor mass.
  RealD CharmNaikEpsilon = calcNaikEpsilon(Action.CharmFermionMass);
  StaggeredFermionOperator LightOp(
    *GridPtr, *GridRBPtr, 2.0 * Action.LightFermionMass, 2.0, -1.0 / 12.0, 1.0, Params
  );
  StaggeredFermionOperator StrangeOp(
    *GridPtr, *GridRBPtr, 2.0 * Action.StrangeFermionMass, 2.0, -1.0 / 12.0, 1.0, Params
  );
  StaggeredFermionOperator RegulatorOp(
    *GridPtr, *GridRBPtr, 2.0 * Action.RegulatorFermionMass, 2.0, -1.0 / 12.0, 1.0, Params
  );
  StaggeredFermionOperator CharmOp(
    *GridPtr, *GridRBPtr, 2.0 * Action.CharmFermionMass, 2.0, -(1.0 + CharmNaikEpsilon) / 12.0, 1.0, Params
  );

  HISQConfiguration<HMCWrapper::ImplPolicy> Policy(GridPtr);
  Primals<HMCWrapper::Field> LightStrangeLinks = Policy.promise();
  Primals<HMCWrapper::Field> CharmLinks = Policy.promise(CharmNaikEpsilon);

  ////////////////////////////////////////////////////////////////
  // Full action
  ////////////////////////////////////////////////////////////////
  ActionLevel<HMCWrapper::Field> Level1(HamiltonianMonteCarlo.FermionActionStepMultiplicity);
  ActionLevel<HMCWrapper::Field> Level2(HamiltonianMonteCarlo.GaugeActionStepMultiplicity);

  auto MILCRationalParams = [&](
    StaggeredFermionOperator& Op,
    int flavors,
    int actionDegree,
    int forceDegree
  ) {
    const RealD m2 = Op.Mass() * Op.Mass();
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
  //        handed to StaggeredEvenEvenRational (heatbath, action, force)
  //   2-4  det(Q_r)^(1/4) each, cancelling the regulator denominator
  //   5    det(Q_c)^(1/4) on the Naik-epsilon links
  std::vector<Factor> LightStrangeFactors = {{LightOp, 2}, {StrangeOp, 1}, {RegulatorOp, -3}};
  StaggeredRationalActionParams LightStrangeParams = MILCRationalParams(
    LightOp, LightStrangeFactors[0].Flavors,
    RationalParams.LightStrangeActionDegree, RationalParams.LightStrangeForceDegree
  );
  StaggeredRationalActionParams RegulatorParams = MILCRationalParams(
    RegulatorOp, 1, RationalParams.RegulatorActionDegree, RationalParams.RegulatorForceDegree
  );
  StaggeredRationalActionParams CharmParams = MILCRationalParams(
    CharmOp, 1, RationalParams.CharmActionDegree, RationalParams.CharmForceDegree
  );

  FermionAction LightStrangePF(
    LightOp, LightStrangeParams,
    combinedRationalFunction(LightStrangeFactors, LightStrangeParams, 8, true, false),
    combinedRationalFunction(LightStrangeFactors, LightStrangeParams, 4, true, true),
    combinedRationalFunction(LightStrangeFactors, LightStrangeParams, 4, false, true)
  );
  FermionAction RegulatorPF1(RegulatorOp, RegulatorParams);
  FermionAction RegulatorPF2(RegulatorOp, RegulatorParams);
  FermionAction RegulatorPF3(RegulatorOp, RegulatorParams);
  FermionAction CharmPF(CharmOp, CharmParams);

  // Eq. A2, beta = 10/g^2. Tadpole improvement is gauge-only.
  // MILC's total_dyn_flavors: the sea quarks l, s, c; the regulator determinants cancel.
  const int nf = LightStrangeFactors[0].Flavors + LightStrangeFactors[1].Flavors + CharmParams.nf;
  RealD u0 = Action.TadpoleFactor;
  PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl> GaugeAction(
    GridPtr, 
    Action.BareGaugeCoupling, 
    1.0, 
    oneLoopMILCRectangleCoefficient(u0, nf), 
    oneLoopMILCParallelogramCoefficient(u0, nf)
  );

  LightStrangePF.signContract(LightStrangeLinks);
  RegulatorPF1.signContract(LightStrangeLinks);
  RegulatorPF2.signContract(LightStrangeLinks);
  RegulatorPF3.signContract(LightStrangeLinks);
  CharmPF.signContract(CharmLinks);

  Level1.push_back(&LightStrangePF);
  Level1.push_back(&RegulatorPF1);
  Level1.push_back(&RegulatorPF2);
  Level1.push_back(&RegulatorPF3);
  Level1.push_back(&CharmPF);
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
