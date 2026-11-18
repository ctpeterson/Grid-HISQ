/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/solver/Test_solver_HISQDirichletBenchmarkHot.cc

Copyright (C) 2015-2016

Author: Alejandro Salas

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
 * @file Test_solver_HISQDirichletBenchmarkHot.cc
 * @author Alejandro Salas
 * @brief Same benchmark as Test_solver_HISQDirichletBenchmarkMILC.cc (full vs
 *        Dirichlet HISQ CG), but on hot-start gauge fields instead of saved
 *        configurations: no files are read.
 *
 * For each of NumberOfConfigurations "configurations" a fresh hot (random SU(3))
 * gauge field is generated with SU<Nc>::HotConfiguration from the parallel RNG
 * (seeded by ParallelSeeds), and then smeared and solved exactly as in the other
 * drivers. ConfigurationPrefix/Suffix/Format, StartConfiguration and
 * ConfigurationInterval are read for compatibility with the same XML files but
 * ignored. Any --grid works (no file dimensions to match).
 *
 * The hot field and the chi sources come from the same parallel RNG, in the same
 * order regardless of --operator, so --operator full and --operator diri runs with
 * the same --grid/--mpi/--xml still solve against identical fields and sources.
 * The source sequence differs from the file-reading drivers, since the hot start
 * consumes random numbers first.
 *
 * A hot field is maximally disordered (plaquette ~ 0), so CG iteration counts and
 * the operator spectrum are not those of an equilibrium ensemble; use it to exercise
 * and time the code path, not as a substitute for physics configurations.
 *
 * Everything else (operators, MILC normalization, Dirichlet geometry, RNG seeding,
 * trace regions, XML) is identical to Test_solver_HISQDirichletBenchmark.cc:
 *
 * Communication-vs-compute profiling benchmark for the HISQ (improved
 * staggered) Dirac operator, comparing the full (undecomposed) operator M
 * against the Dirichlet-truncated (domain-decomposed) operator V used by
 * DDHMC. Reads in one or more saved thin-link configurations, applies the
 * same HISQ smearing the production HISQ action uses (HISQConfiguration:
 * fat7 -> unitary projection -> asqtad + Lepage), then repeatedly solves
 * psi = (Op^dag Op)^-1 chi with plain CG against EITHER M or V (selected via
 * --operator full|diri on the command line -- never both in the same run,
 * so an external profiler attributes all of a run's communication
 * unambiguously to the one operator).
 *
 * The operator normalization matches HMC/HISQ2p1p1f.cc (MILC convention,
 * M = 2m + Dslash): mass 2m, one-link coefficient 2, Naik coefficient
 * -(1 + epsilon)/12, u0 = 1. The Naik epsilon is zero for light and
 * strange quarks; set UseNaikEpsilon for charm, in which case epsilon is
 * computed from the bare mass with calcNaikEpsilon, exactly as in
 * HISQ2p1p1f.cc. The normalization does not change CG iteration counts
 * (the relative residual is invariant under Q -> 4Q).
 *
 * The HISQ smearing runs on the full periodic lattice (so it communicates
 * in both --operator modes) and is kept in its own trace region, separate
 * from the solve. Dirichlet masks are applied by ImportGauge AFTER smearing,
 * with depth one for the one-link term and three for the Naik term.
 *
 * chi is drawn fresh (Gaussian, even-checkerboard) for every repeat, but
 * the draw itself does not depend on --operator, so running this program
 * twice with the same --grid/--mpi/--xml and --operator full vs --operator
 * diri reproduces the EXACT SAME sequence of chi's for both -- a paired
 * comparison, not two independent ones.
 *
 * No HMC integration, no momentum, no Metropolis step -- this only
 * exercises the CG solve itself, so it belongs in tests/solver/, not
 * tests/hmc/.
 */

#include <Grid/Grid.h>

using namespace Grid;

////////////////////////////////////////////////////////////////
// Typedefs
////////////////////////////////////////////////////////////////
typedef ImprovedStaggeredFermionD StaggeredFermionOperator;
typedef typename StaggeredFermionOperator::FermionField FermionField;
typedef StaggeredImplD FermionImplPolicy;
typedef PeriodicGimplD Gimpl;
typedef typename Gimpl::Field GaugeField;

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

GRID_SERIALIZABLE(
  ActionParameters,
  double, Mass,
  bool, UseNaikEpsilon
);

GRID_SERIALIZABLE(
  RandomNumberGeneratorParameters,
  std::string, SerialSeeds,
  std::string, ParallelSeeds
);

GRID_SERIALIZABLE(
  BenchmarkParameters,
  std::string, ConfigurationPrefix,  // ignored in this driver (hot start; kept so the same XML works)
  std::string, ConfigurationSuffix,  // ignored
  std::string, ConfigurationFormat,  // ignored
  int, StartConfiguration,           // ignored
  int, ConfigurationInterval,        // ignored
  int, NumberOfConfigurations,
  int, RepeatsPerConfiguration
);

GRID_SERIALIZABLE(
  CGParameters,
  double, StoppingCondition,
  int, MaxIterations
);

#undef GRID_SERIALIZABLE

std::vector<int> parseSeeds(const std::string& s) {
  std::vector<int> seeds;
  std::stringstream ss(s);
  int x;
  while (ss >> x) { seeds.push_back(x); }
  return seeds;
}

// Physical checks of a configuration (for a hot start: plaquette ~ 0, unitary links)
void reportGaugeChecks(const GaugeField& U) {
  RealD plaq = WilsonLoops<Gimpl>::avgPlaquette(U);
  std::vector<RealD> ts = WilsonLoops<Gimpl>::timesliceAvgSpatialPlaquette(U);
  RealD ssPlaq = 0.0;
  for (auto p : ts) { ssPlaq += p; }
  ssPlaq /= ts.size();
  RealD stPlaq = 2.0 * plaq - ssPlaq;  // average of 3 spatial and 3 temporal planes
  RealD ltr = WilsonLoops<Gimpl>::linkTrace(U);

  LatticeColourMatrixD ident(U.Grid());
  LatticeColourMatrixD Umu(U.Grid());
  ident = 1.0;
  RealD defect = 0.0;
  for (int mu = 0; mu < Nd; ++mu) {
    Umu = PeekIndex<LorentzIndex>(U, mu);
    defect += norm2(adj(Umu) * Umu - ident);
  }
  defect = std::sqrt(defect / (U.Grid()->gSites() * (RealD)Nd));

  std::cout << GridLogMessage << "BENCHMARK gauge checks: plaquette " << plaq
             << " (ss " << ssPlaq << " st " << stPlaq << ")"
             << " MILC-normalized PLAQ (x Nc): " << Nc * ssPlaq << " " << Nc * stPlaq
             << " link trace " << ltr
             << " rms unitarity defect " << defect << std::endl;
  if (defect > 1.0e-4) {
    std::cout << GridLogWarning << "WARNING: links are far from unitary -- the data was "
               << "probably read with the wrong byte order or layout" << std::endl;
  }
}

////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  Serialiser Reader(GridCmdOptionPayload(argv, argv + argc, "--xml"), false, "grid");

  ActionParameters Action(Reader);
  RandomNumberGeneratorParameters RandomNumberGenerator(Reader);
  BenchmarkParameters Benchmark(Reader);
  CGParameters Solver(Reader);

  std::string operatorChoice = GridCmdOptionPayload(argv, argv + argc, "--operator");
  bool useFull;
  if (operatorChoice == std::string("full")) { useFull = true; }
  else if (operatorChoice == std::string("diri")) { useFull = false; }
  else {
    std::cout << GridLogError << "ERROR: --operator must be 'full' or 'diri' (got '"
               << operatorChoice << "')" << std::endl;
    Grid_finalize();
    return 1;
  }

  ////////////////////////////////////////////////////////////////
  // Grid setup -- no HMC wrapper, this only exercises the solver
  ////////////////////////////////////////////////////////////////
  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(
                                      GridDefaultLatt(),
                                      GridDefaultSimd(Nd, vComplexD::Nsimd()),
                                      GridDefaultMpi());
  GridRedBlackCartesian  *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  ////////////////////////////////////////////////////////////////
  // RNG -- seeded directly from the XML (not via HMC resources), so the
  // same chi sequence is reproduced across --operator full/diri runs as
  // long as --grid/--mpi and these seeds match between them.
  ////////////////////////////////////////////////////////////////
  GridSerialRNG   sRNG;
  sRNG.SeedFixedIntegers(parseSeeds(RandomNumberGenerator.SerialSeeds));
  GridParallelRNG pRNG(UGrid);
  pRNG.SeedFixedIntegers(parseSeeds(RandomNumberGenerator.ParallelSeeds));

  ////////////////////////////////////////////////////////////////
  // Dirichlet block (same geometry rule as the DDHMC test): one block
  // per node, cut only in directions split across more than one node.
  // FullOp always gets the all-zero block; DiriOp gets the
  // domain-decomposed one.
  ////////////////////////////////////////////////////////////////
  Coordinate latt_size  = UGrid->GlobalDimensions();
  Coordinate mpi_layout = GridDefaultMpi();
  Coordinate shm;
  Coordinate CommDim(Nd);
  Coordinate NonDirichlet(Nd, 0);
  Coordinate Dirichlet(Nd, 0);

  GlobalSharedMemory::GetShmDims(mpi_layout, shm);
  for (int mu = 0; mu < Nd; ++mu) {
    CommDim[mu] = (mpi_layout[mu] / shm[mu]) > 1 ? 1 : 0;
    Dirichlet[mu] = CommDim[mu] * latt_size[mu] / mpi_layout[mu] * shm[mu];
    std::cout << GridLogMessage << "Dirichlet BCs in direction " << mu
               << " = " << Dirichlet[mu] << std::endl;
  }

  std::vector<Complex> boundary = {1, 1, 1, -1};
  StaggeredFermionOperator::ImplParams Full(boundary);
  StaggeredFermionOperator::ImplParams Diri(boundary);
  Full.dirichlet = NonDirichlet;
  Diri.dirichlet = Dirichlet;

  ////////////////////////////////////////////////////////////////
  // HISQ operators -- MILC normalization as in HMC/HISQ2p1p1f.cc.
  // Naik epsilon uses the bare mass, not the doubled constructor mass.
  ////////////////////////////////////////////////////////////////
  RealD NaikEpsilon = Action.UseNaikEpsilon ? calcNaikEpsilon(Action.Mass) : 0.0;
  std::cout << GridLogMessage << "Bare mass " << Action.Mass
             << " Naik epsilon " << NaikEpsilon << std::endl;

  StaggeredFermionOperator FullOp(
    *UGrid, *UrbGrid, 2.0 * Action.Mass, 2.0, -(1.0 + NaikEpsilon) / 12.0, 1.0, Full
  );
  StaggeredFermionOperator DiriOp(
    *UGrid, *UrbGrid, 2.0 * Action.Mass, 2.0, -(1.0 + NaikEpsilon) / 12.0, 1.0, Diri
  );

  ////////////////////////////////////////////////////////////////
  // HISQ smearing (applied once per configuration, identically for both
  // operators -- matches how the production action shares one smeared
  // field between V and M). links[0] is the unitarized fat7 field W
  // (input to the Naik term); links[1] is the asqtad field X + (eps/8) W
  // (input to the one-link term).
  ////////////////////////////////////////////////////////////////
  HISQConfiguration<Gimpl> Policy(UGrid);
  Primals<GaugeField> links = Policy.promise(NaikEpsilon);

  GaugeField Uthin(UGrid);
  GaugeField Unaik(UGrid);
  GaugeField Ufat(UGrid);

  ConjugateGradient<FermionField> CG(Solver.StoppingCondition, Solver.MaxIterations, false);

  for (int c = 0; c < Benchmark.NumberOfConfigurations; ++c) {
    int trajNum = c;  // label of the c-th hot configuration

    {
      GRID_TRACE("hot_start");
      SU<Nc>::HotConfiguration(pRNG, Uthin);
      std::cout << GridLogMessage << "BENCHMARK hot configuration " << trajNum
                 << " plaquette " << WilsonLoops<Gimpl>::avgPlaquette(Uthin) << std::endl;
      reportGaugeChecks(Uthin);
    }

    {
      GRID_TRACE("hisq_smear");
      Policy.set_Field(Uthin);
      Unaik = links[0].resolve();
      Ufat  = links[1].resolve();
    }

    {
      GRID_TRACE("import_gauge");
      if (useFull) { FullOp.ImportGauge(Unaik, Ufat); }
      else         { DiriOp.ImportGauge(Unaik, Ufat); }
    }

    for (int r = 0; r < Benchmark.RepeatsPerConfiguration; ++r) {
      FermionField eta(UGrid);
      FermionField chi(UrbGrid);
      FermionField psi(UrbGrid);

      // Drawn unconditionally, before the operator-specific branch below,
      // so this call happens identically regardless of --operator.
      gaussian(pRNG, eta);
      pickCheckerboard(Even, chi, eta);
      psi.Checkerboard() = Even;
      psi = Zero();

      std::cout << GridLogMessage << "BENCHMARK config=" << trajNum
                 << " repeat=" << r << " operator=" << operatorChoice << std::endl;

      if (useFull) {
        GRID_TRACE("solve_full");
        SchurStaggeredOperator<StaggeredFermionOperator, FermionField> Mop(FullOp);
        CG(Mop, chi, psi);
      } else {
        GRID_TRACE("solve_diri");
        SchurStaggeredOperator<StaggeredFermionOperator, FermionField> Mop(DiriOp);
        CG(Mop, chi, psi);
      }
    }
  }

  Grid_finalize();
}

/* Example XML input file (pass as --xml <xml-file-name>.xml), and pass
   --operator full or --operator diri on the command line. The XML is the
   same as for the file-reading drivers; the configuration-file fields
   (ConfigurationPrefix/Suffix/Format, StartConfiguration, ConfigurationInterval)
   are ignored here and may be left out (Grid then only prints a warning).
   E.g. with the 32^3 x 48 geometry used so far (run with --grid 32.32.32.48):

<?xml version="1.0"?>
<grid>
  <ActionParameters>
    <Mass>0.0673</Mass>
    <UseNaikEpsilon>false</UseNaikEpsilon>
  </ActionParameters>
  <RandomNumberGeneratorParameters>
    <SerialSeeds>1 2 3 4 5</SerialSeeds>
    <ParallelSeeds>6 7 8 9 10</ParallelSeeds>
  </RandomNumberGeneratorParameters>
  <BenchmarkParameters>
    <ConfigurationPrefix>/mnt/ffs24/home/pete1776/ddhmc/configs/0p15/l3248f211b580m002426m06730m8447a</ConfigurationPrefix>
    <ConfigurationSuffix>.ildg</ConfigurationSuffix>
    <ConfigurationFormat>ildg</ConfigurationFormat>
    <StartConfiguration>1000</StartConfiguration>
    <ConfigurationInterval>100</ConfigurationInterval>
    <NumberOfConfigurations>1</NumberOfConfigurations>
    <RepeatsPerConfiguration>2</RepeatsPerConfiguration>
  </BenchmarkParameters>
  <CGParameters>
    <StoppingCondition>1e-8</StoppingCondition>
    <MaxIterations>5000</MaxIterations>
  </CGParameters>
</grid>

This generates 1 hot configuration and does 2 solves against it. */
