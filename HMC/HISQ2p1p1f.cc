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
typedef TwoPlusOneFlavorStaggeredEvenEvenRatioRational<FermionImplPolicy> LightStrangeAction;

typedef PeriodicGimplD Gimpl;

typedef XmlReader Serialiser;

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

GRID_SERIALIZABLE(
  RationalApproximationParameters,
  double, LightLowerBound,
  double, LightUpperBound,
  double, StrangeLowerBound,
  double, StrangeUpperBound,
  double, CharmLowerBound,
  double, CharmUpperBound,
  double, RegulatorLowerBound,
  double, RegulatorUpperBound,
  int, ActionDegree,
  int, ForceDegree,
  int, Precision
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
  // XML masses remain bare am; rational bounds refer to this MdagM (=4Q).
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
  LinkBinding<HMCWrapper::Field> LightStrangeLinks = Policy.links();
  LinkBinding<HMCWrapper::Field> CharmLinks = Policy.links(CharmNaikEpsilon);

  ////////////////////////////////////////////////////////////////
  // Full action
  ////////////////////////////////////////////////////////////////
  ActionLevel<HMCWrapper::Field> Level1(HamiltonianMonteCarlo.FermionActionStepMultiplicity);
  ActionLevel<HMCWrapper::Field> Level2(HamiltonianMonteCarlo.GaugeActionStepMultiplicity);

  // For Q = MdagM on the even grid, the weights are det(Q)^(nf/4).
  // Bounds must cover the even-grid normal-operator spectrum for each mass.
  StaggeredRationalActionParams LightParams(
    2, 
    RationalParams.LightLowerBound, 
    RationalParams.LightUpperBound,
    CGParams.MaxIterations, 
    CGParams.ActionStoppingCondition, 
    RationalParams.ActionDegree,
    CGParams.ForceStoppingCondition, 
    RationalParams.ForceDegree, 
    RationalParams.Precision, 
    0
  );
  StaggeredRationalActionParams StrangeParams(
    1, 
    RationalParams.StrangeLowerBound, 
    RationalParams.StrangeUpperBound,
    CGParams.MaxIterations, 
    CGParams.ActionStoppingCondition, 
    RationalParams.ActionDegree,
    CGParams.ForceStoppingCondition, 
    RationalParams.ForceDegree, 
    RationalParams.Precision, 
    0
  );
  StaggeredRationalActionParams CharmParams(
    1, 
    RationalParams.CharmLowerBound, 
    RationalParams.CharmUpperBound,
    CGParams.MaxIterations, 
    CGParams.ActionStoppingCondition, 
    RationalParams.ActionDegree,
    CGParams.ForceStoppingCondition, 
    RationalParams.ForceDegree, 
    RationalParams.Precision, 
    0
  );

  StaggeredRationalActionParams RegulatorParams(
    1,
    RationalParams.RegulatorLowerBound,
    RationalParams.RegulatorUpperBound,
    CGParams.MaxIterations,
    CGParams.ActionStoppingCondition,
    RationalParams.ActionDegree,
    CGParams.ForceStoppingCondition,
    RationalParams.ForceDegree,
    RationalParams.Precision,
    0
  );
  StaggeredRationalActionParams RegulatorRatioParams = RegulatorParams;
  RegulatorRatioParams.nf = 3;

  // The combined weight is det(Q_l)^(1/2) det(Q_s)^(1/4) / det(Q_r)^(3/4).
  // Three independent det(Q_r)^(1/4) fields cancel the regulator denominator.
  LightStrangeAction LightStrangePF(
    RegulatorOp, LightOp, StrangeOp, RegulatorRatioParams, LightParams, StrangeParams
  );
  FermionAction RegulatorPF1(RegulatorOp, RegulatorParams);
  FermionAction RegulatorPF2(RegulatorOp, RegulatorParams);
  FermionAction RegulatorPF3(RegulatorOp, RegulatorParams);
  FermionAction CharmPF(CharmOp, CharmParams);

  // Eq. A2, nf = 4 and beta = 10/g^2. Tadpole improvement is gauge-only.
  RealD u0 = Action.TadpoleFactor;
  RealD RectangleCoefficient = oneLoopMILCRectangleCoefficient(u0);
  RealD ParallelogramCoefficient = oneLoopMILCParallelogramCoefficient(u0);
  PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl> GaugeAction(
    GridPtr, Action.BareGaugeCoupling, 1.0, RectangleCoefficient, ParallelogramCoefficient
  );

  LightStrangePF.bindLinks(LightStrangeLinks, LightStrangeLinks, LightStrangeLinks);
  RegulatorPF1.bindLinks(LightStrangeLinks);
  RegulatorPF2.bindLinks(LightStrangeLinks);
  RegulatorPF3.bindLinks(LightStrangeLinks);
  CharmPF.bindLinks(CharmLinks);

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
   Phys. Rev. D 82, 074501. Step counts and rational settings below are examples,
   not a reproduction of a MILC production input deck; check them before use.
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
    <SerialSeeds>1 2 3 4 5</SerialSeeds>
    <ParallelSeeds>6 7 8 9 10</ParallelSeeds>
  </RandomNumberGeneratorParameters>
  <ConjugateGradientParameters>
    <ActionStoppingCondition>1e-10</ActionStoppingCondition>
    <ForceStoppingCondition>1e-8</ForceStoppingCondition>
    <MaxIterations>10000</MaxIterations>
  </ConjugateGradientParameters>
  <RationalApproximationParameters>
    <LightLowerBound>0.000012</LightLowerBound>
    <LightUpperBound>256.0</LightUpperBound>
    <StrangeLowerBound>0.01</StrangeLowerBound>
    <StrangeUpperBound>256.0</StrangeUpperBound>
    <CharmLowerBound>1.56</CharmLowerBound>
    <CharmUpperBound>256.0</CharmUpperBound>
    <RegulatorLowerBound>0.156</RegulatorLowerBound>
    <RegulatorUpperBound>256.0</RegulatorUpperBound>
    <ActionDegree>40</ActionDegree>
    <ForceDegree>32</ForceDegree>
    <Precision>80</Precision>
  </RationalApproximationParameters>
</grid>
*/
