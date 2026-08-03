/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/hmc/Test_hmc_DDStaggeredFourFlavor.cc

Copyright (C) 2015-2016

Author: Alejandro Salas
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
 * @file Test_hmc_DDStaggeredFourFlavor.cc
 * @author Alejandro Salas
 * @author Curtis Taylor Peterson
 * @brief Domain-decomposed HMC for stout-smeared naive staggered fermion
 */

#include <Grid/Grid.h>

using namespace Grid;

////////////////////////////////////////////////////////////////
// Typedefs
////////////////////////////////////////////////////////////////
typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;

typedef NaiveStaggeredFermionD StaggeredFermionOperator;
typedef typename StaggeredFermionOperator::FermionField FermionField;
typedef StaggeredImplD FermionImplPolicy;
typedef FourFlavorStaggeredEvenEvenRatioPseudoFermionAction<FermionImplPolicy> BoundaryAction;
typedef FourFlavorStaggeredEvenEvenPseudoFermionAction<FermionImplPolicy> LocalAction;

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
  
  int, BoundaryActionStepMultiplicity,
  int, LocalActionStepMultiplicity,
  int, GaugeActionStepMultiplicity
);

GRID_SERIALIZABLE(
  DomainDecompositionParameters,
  bool, DomainDecomposed,
  int, MomentumFilterWidth
);

GRID_SERIALIZABLE(
  ActionParameters,
  double, BareGaugeCoupling,
  double, FermionMass,

  bool, StoutSmear,
  double, StoutSmearRho,
  int, StoutSmearSteps
);

GRID_SERIALIZABLE(
  ConjugateGradientParameters,
  double, ActionStoppingCondition,
  double, ActionMaxIterations,
  double, ForceStoppingCondition,
  double, ForceMaxIterations
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
  DomainDecompositionParameters DomainDecomposition(Reader);
  ActionParameters Action(Reader);
  CheckpointParameters Checkpoint(Reader);
  RandomNumberGeneratorParameters RandomNumberGenerator(Reader);
  ConjugateGradientParameters CGParams(Reader);

  IntegratorParameters MDParams;
  HMCparameters HMCParams;
  CheckpointerParameters CPParams;
  RNGModuleParameters RNGParams;

  MDParams.name = std::string("MinimumNorm2");
  MDParams.MDsteps = HamiltonianMonteCarlo.MolecularDynamicsSteps;
  MDParams.trajL = HamiltonianMonteCarlo.TrajectoryLength;

  HMCParams.NoMetropolisUntil = HamiltonianMonteCarlo.NoMetropolisUntil;
  HMCParams.StartingType = Initialization.StartType;
  HMCParams.PerformRandomShift = DomainDecomposition.DomainDecomposed;
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
  HMCWrapper TheHMC;

  TheHMC.ReadCommandLine(argc, argv);
  TheHMC.Parameters = HMCParams;
  TheHMC.Resources.AddFourDimGrid("gauge");
  TheHMC.Resources.LoadNerscCheckpointer(CPParams);
  TheHMC.Resources.SetRNGSeeds(RNGParams);

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();

  ////////////////////////////////////////////////////////////////
  // Domain decomposition, Dirichlet BCs, and momentum filter
  ////////////////////////////////////////////////////////////////
  Coordinate latt_size  = GridPtr->GlobalDimensions();
  Coordinate mpi_layout = GridDefaultMpi(); // Check this
  
  Coordinate shm;
  Coordinate CommDim(Nd);
  Coordinate NonDirichlet(Nd, 0);
  Coordinate Dirichlet(Nd, 0);
  Coordinate Block4(Nd);

  if (DomainDecomposition.DomainDecomposed) {
    GlobalSharedMemory::GetShmDims(mpi_layout, shm);
    for (int mu = 0; mu < Nd; ++mu) { 
      CommDim[mu] = (mpi_layout[mu] / shm[mu]) > 1 ? 1 : 0;
      Dirichlet[mu] = CommDim[mu] * latt_size[mu] / mpi_layout[mu] * shm[mu];
      Block4[mu] = Dirichlet[mu];
      std::cout << GridLogMessage 
                << " Dirichlet BCs in direction " 
                << mu << " = " << Dirichlet[mu] << std::endl;
    }

    TheHMC.Resources.SetMomentumFilter(
      new DDHMCFilter<WilsonImplR::Field>(Block4, DomainDecomposition.MomentumFilterWidth)
    );
  }

  ////////////////////////////////////////////////////////////////
  // Fermion operators
  ////////////////////////////////////////////////////////////////
  LatticeGaugeField U(GridPtr);

  std::vector<Complex> boundary = {1, 1, 1, -1};
  StaggeredFermionOperator::ImplParams Full(boundary);
  StaggeredFermionOperator::ImplParams Diri(boundary);
  
  Full.dirichlet = NonDirichlet;
  if (DomainDecomposition.DomainDecomposed) { Diri.dirichlet = Dirichlet; } 
  else { Diri.dirichlet = NonDirichlet; }

  StaggeredFermionOperator FullOp(*GridPtr, *GridRBPtr, Action.FermionMass, 1.0, 1.0, Full);
  StaggeredFermionOperator DiriOp(*GridPtr, *GridRBPtr, Action.FermionMass, 1.0, 1.0, Diri);

  ////////////////////////////////////////////////////////////////
  // Full action
  ////////////////////////////////////////////////////////////////
  ActionLevel<HMCWrapper::Field> Level1(HamiltonianMonteCarlo.BoundaryActionStepMultiplicity); 
  ActionLevel<HMCWrapper::Field> Level2(HamiltonianMonteCarlo.LocalActionStepMultiplicity);
  ActionLevel<HMCWrapper::Field> Level3(HamiltonianMonteCarlo.GaugeActionStepMultiplicity);

  ConjugateGradient<FermionField> ForceCG(
    CGParams.ForceStoppingCondition,
    CGParams.ForceMaxIterations
  );
  ConjugateGradient<FermionField> ActionCG(
    CGParams.ActionStoppingCondition,
    CGParams.ActionMaxIterations
  );

  BoundaryAction BoundaryPF(DiriOp, FullOp, ForceCG, ActionCG);
  LocalAction LocalPF(DiriOp, ForceCG, ActionCG);
  WilsonGaugeActionR GaugeAction(Action.BareGaugeCoupling);

  BoundaryPF.is_smeared = Action.StoutSmear;
  LocalPF.is_smeared = Action.StoutSmear;
  
  if (DomainDecomposition.DomainDecomposed) { Level1.push_back(&BoundaryPF); }
  Level2.push_back(&LocalPF);
  Level3.push_back(&GaugeAction);

  if (DomainDecomposition.DomainDecomposed) { TheHMC.TheAction.push_back(Level1); }
  TheHMC.TheAction.push_back(Level2);
  TheHMC.TheAction.push_back(Level3);

  //////////////////////////////////////////////////////////////
  // Run HMC with or without stout smearing
  //////////////////////////////////////////////////////////////
  TheHMC.Resources.AddObservable<PlaqObs>();
  if (Action.StoutSmear) { 
    Smear_Stout<HMCWrapper::ImplPolicy> Stout(Action.StoutSmearRho);
    SmearedConfiguration<HMCWrapper::ImplPolicy> Policy(GridPtr, Action.StoutSmearSteps, Stout);
    TheHMC.Run(Policy);
  } else { TheHMC.Run(); }

  Grid_finalize();
}

/* Example XML input file (pass as --xml <xml-file-name>.xml):
<?xml version="1.0"?>
<grid>
  <InitializationParameters>
    <StartType>TepidStart</StartType>
  </InitializationParameters>
  <HamiltonianMonteCarloParameters>
    <NoMetropolisUntil>0</NoMetropolisUntil>
    <TrajectoryLength>1.0</TrajectoryLength>
    <MolecularDynamicsSteps>15</MolecularDynamicsSteps>
    <BoundaryActionStepMultiplicity>1</BoundaryActionStepMultiplicity>
    <LocalActionStepMultiplicity>1</LocalActionStepMultiplicity>
    <GaugeActionStepMultiplicity>1</GaugeActionStepMultiplicity>
  </HamiltonianMonteCarloParameters>
  <DomainDecompositionParameters>
    <DomainDecomposed>true</DomainDecomposed>
    <MomentumFilterWidth>3</MomentumFilterWidth>
  </DomainDecompositionParameters>
  <ActionParameters>
    <BareGaugeCoupling>9.0</BareGaugeCoupling>
    <FermionMass>0.01</FermionMass>
    <StoutSmear>true</StoutSmear>
    <StoutSmearRho>0.1</StoutSmearRho>
    <StoutSmearSteps>3</StoutSmearSteps>
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
    <ActionMaxIterations>10000</ActionMaxIterations>
    <ForceStoppingCondition>1e-8</ForceStoppingCondition>
    <ForceMaxIterations>10000</ForceMaxIterations>
  </ConjugateGradientParameters>
</grid>
*/
