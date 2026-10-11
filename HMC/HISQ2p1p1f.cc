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

#ifdef GRID_MILC_RNG_AUDIT
#include <Grid/GridCore.h>
#include <Grid/qcd/QCD.h>
namespace MilcAudit {
void checkNoise(Grid::GridParallelRNG &,const Grid::LatticeColourVectorD &);
}
namespace Grid {
// Observe the actual noise produced inside the pseudofermion refresh routines.
inline void gaussian(GridParallelRNG &rng,LatticeColourVectorD &field) {
  rng.fill(field,rng._gaussian);
  MilcAudit::checkNoise(rng,field);
}
}
#endif
#include <Grid/Grid.h>
#ifdef GRID_MILC_RNG_AUDIT
// Optional validation against unmodified MILC C routines. Build with
// MILC_PRECISION=2 and link ranstuff.c, gaussrand.c, and rand_ahmat.c.
namespace MilcReference {
extern "C" {
#include "../milc_qcd/include/su3.h"
void initialize_prn(double_prn *, uint32_t, uint32_t);
}
}
namespace MilcAudit {
using namespace Grid;
GridCartesian *grid = nullptr;
std::vector<MilcReference::double_prn> sites;
MilcReference::double_prn serial{};
std::string prefix;
std::vector<const LatticeColourVectorD *> refreshedFields;
int trajectory = 1, noises = 0;

std::string filename(const std::string &stage) {
  return prefix + "." + std::to_string(trajectory) + ".rank" +
         std::to_string(grid->ThisRank()) + "." + stage;
}
void require(bool ok, const char *message) {
  uint64_t failures = !ok;
  grid->GlobalSum(failures);
  if (failures) throw std::runtime_error(std::string("MILC RNG AUDIT: ") + message);
}
bool same(const GridRNGbase::RngEngine &engine, const MilcReference::double_prn &ref) {
  auto s = engine.state();
  const unsigned long r[] = {ref.r0,ref.r1,ref.r2,ref.r3,ref.r4,ref.r5,ref.r6};
  for (int j=0;j<7;j++) if(s[j+2]!=r[j]) return false;
  return s[9]==uint32_t(ref.ic_state) && s[10]==uint32_t(ref.ic_state>>32) && s[11]==ref.multiplier;
}
void checkStates(GridParallelRNG &rng) {
  bool ok=true;
  for(int i=0;i<grid->lSites();i++) {
    Coordinate c; grid->LocalIndexToLocalCoor(i,c);
    int slot=rng.generator_idx(grid->oIndex(c),grid->iIndex(c));
    ok &= same(rng._generators[slot],sites[i]);
  }
  require(ok,"site engine state mismatch");
}
void checkSerial(GridSerialRNG &rng) {
  require(same(rng._generators[0],serial),"serial state differs from MILC rank zero");
  GridSerialRNG probe; probe._generators=rng._generators; probe._uniform=rng._uniform;
  auto ref=serial;
  bool ok=true;
  std::ofstream out(filename("serial-probe"),std::ios::binary);
  for(int i=0;i<32;i++) {
    double actual; random(probe,actual);
    double expected=MilcReference::myrand(&ref);
    ok &= actual==expected;
    out.write(reinterpret_cast<const char*>(&actual),sizeof(actual));
  }
  require(ok,"serial uniform continuation mismatch");
}
template<class V> void compare(const Lattice<V> &actual,const Lattice<V> &expected,const std::string &stage) {
  double rel=std::sqrt(norm2(actual-expected)/std::max(1.0,norm2(expected)));
  std::vector<typename V::scalar_object> values;
  unvectorizeToLexOrdArray(values,actual);
  std::ofstream out(filename(stage+".bin"),std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));
  require(bool(out),"cannot write draw comparison artifact");
  if(grid->IsBoss()) std::cout << GridLogMessage << "MILC_AUDIT trajectory=" << trajectory
    << " stage=" << stage << " relative_L2=" << std::setprecision(17) << rel << std::endl;
  require(rel<1e-13,"Gaussian field differs from MILC");
}
void initialize(GridCartesian *g, const std::string &seedText, const std::string &serialText,
                const std::string &filePrefix, int start, bool restart) {
  grid=g; prefix=filePrefix; trajectory=start+1;
  auto seed=strToVec<int>(seedText), serialSeed=strToVec<int>(serialText);
  require(seed.size()==1 && serialSeed.size()==2,"use one parallel seed and {seed, volume} for serial");
  require(seed[0]==serialSeed[0] && uint32_t(serialSeed[1])==grid->gSites(),"serial seed must match MILC {seed, volume}");
  sites.resize(grid->lSites());
  for(int i=0;i<grid->lSites();i++) {
    Coordinate c,global; grid->LocalIndexToLocalCoor(i,c);
    Coordinate proc=grid->ThisProcessorCoor();
    grid->ProcessorCoorLocalCoorToGlobalCoor(proc,c,global);
    uint32_t index=global[0]+grid->_fdimensions[0]*(global[1]+grid->_fdimensions[1]*(global[2]+grid->_fdimensions[2]*global[3]));
    MilcReference::initialize_prn(&sites[i],uint32_t(seed[0]),index);
  }
  MilcReference::initialize_prn(&serial,uint32_t(seed[0]),uint32_t(grid->gSites()));
  if(restart) {
    trajectory=start;
    std::ifstream in(filename("milc-state"),std::ios::binary);
    in.read(reinterpret_cast<char*>(&serial),sizeof(serial));
    in.read(reinterpret_cast<char*>(sites.data()),sites.size()*sizeof(sites[0]));
    require(bool(in),"cannot restore independent MILC reference states");
    trajectory=start+1;
  }
}
void checkNoise(GridParallelRNG &rng,const LatticeColourVectorD &noise) {
  if(!grid) return;
  require(noise.Grid()->_isCheckerBoarded && noise.Checkerboard()==Even,"pseudofermion noise is not even-only");
  LatticeColourVectorD full(grid),expected(noise.Grid()),actual(noise.Grid());
  full=Zero(); expected.Checkerboard()=actual.Checkerboard()=Even;
  for(int i=0;i<grid->lSites();i++) {
    Coordinate c,global; grid->LocalIndexToLocalCoor(i,c);
    Coordinate proc=grid->ThisProcessorCoor();
    grid->ProcessorCoorLocalCoorToGlobalCoor(proc,c,global);
    if((global[0]+global[1]+global[2]+global[3])%2) continue;
    ColourVectorD value;
    for(int j=0;j<3;j++) {
      auto z=MilcReference::complex_gaussian_rand_no(&sites[i]);
      value()()(j)=ComplexD(z.real,z.imag);
    }
    pokeLocalSite(value,full,c);
  }
  pickCheckerboard(Even,expected,full);
  actual=noise*std::sqrt(0.5);
  compare(actual,expected,"eta"+std::to_string(++noises));
  checkStates(rng);
}
void checkMomenta(const LatticeGaugeFieldD &momenta,GridParallelRNG &rng) {
  require(noises==0,"unexpected refresh ordering");
  LatticeGaugeFieldD expected(grid);
  for(int i=0;i<grid->lSites();i++) {
    Coordinate c; grid->LocalIndexToLocalCoor(i,c);
    LorentzColourMatrixD value;
    for(int mu=0;mu<4;mu++) {
      MilcReference::anti_hermitmat m;
      MilcReference::random_anti_hermitian(&m,&sites[i]);
      auto &a=value(mu)();
      a(0,0)=ComplexD(0,m.m00im); a(1,1)=ComplexD(0,m.m11im); a(2,2)=ComplexD(0,m.m22im);
      a(0,1)=ComplexD(m.m01.real,m.m01.imag); a(0,2)=ComplexD(m.m02.real,m.m02.imag);
      a(1,2)=ComplexD(m.m12.real,m.m12.imag);
      a(1,0)=ComplexD(-m.m01.real,m.m01.imag); a(2,0)=ComplexD(-m.m02.real,m.m02.imag);
      a(2,1)=ComplexD(-m.m12.real,m.m12.imag);
    }
    pokeLocalSite(value,expected,c);
  }
  compare(momenta,expected,"momentum");
  checkStates(rng);
}
struct Implementation : PeriodicGimplD {
  static void generate_momenta(Field &p,GridSerialRNG &srng,GridParallelRNG &prng) {
    checkSerial(srng);
    PeriodicGimplD::generate_momenta(p,srng,prng);
    checkMomenta(p,prng);
  }
};
struct Observable : HmcObservable<LatticeGaugeFieldD> {
  using Field=LatticeGaugeFieldD;
  void TrajectoryComplete(int traj,Field &,GridSerialRNG &srng,GridParallelRNG &prng) override {
    require(traj==trajectory && noises==5,"expected five pseudofermion refreshes per trajectory");
    // MILC update_rhmc.c draws once on rank zero for the Metropolis decision.
    for(size_t i=0;i<refreshedFields.size();i++) {
      std::vector<ColourVectorD> values;
      unvectorizeToLexOrdArray(values,*refreshedFields[i]);
      std::ofstream phi(filename("phi"+std::to_string(i+1)+".bin"),std::ios::binary);
      phi.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));
      require(bool(phi),"cannot save refreshed pseudofermion field");
    }
    MilcReference::myrand(&serial);
    checkSerial(srng); checkStates(prng);
    std::ofstream out(filename("milc-state"),std::ios::binary);
    out.write(reinterpret_cast<const char*>(&serial),sizeof(serial));
    out.write(reinterpret_cast<const char*>(sites.data()),sites.size()*sizeof(sites[0]));
    require(bool(out),"cannot save independent MILC reference states");
    if(grid->IsBoss()) std::cout << GridLogMessage << "MILC_AUDIT trajectory=" << traj
      << " PASS all fields, site states, serial state and 32 subsequent serial draws" << std::endl;
    ++trajectory; noises=0;
  }
};
struct Module : ObservableModule<Observable,NoParameters> {
  Module():ObservableModule(NoParameters()){}
  void initialize() override { ObservablePtr.reset(new Observable); }
};
}
#endif


using namespace Grid;

////////////////////////////////////////////////////////////////
// Typedefs
////////////////////////////////////////////////////////////////
#ifdef GRID_MILC_RNG_AUDIT
typedef HMCWrapperTemplate<MilcAudit::Implementation,MinimumNorm2> HMCWrapper;
#else
typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
#endif
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
#ifdef GRID_MILC_RNG_AUDIT
  MilcAudit::initialize(GridPtr, RandomNumberGenerator.ParallelSeeds,
    RandomNumberGenerator.SerialSeeds,
    GridCmdOptionPayload(argv,argv+argc,"--rng-audit-prefix"),
    TheHMC.Parameters.StartTrajectory,
    TheHMC.Parameters.StartingType == "CheckpointStart");
  TheHMC.Resources.AddObservable<MilcAudit::Module>();
#endif

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
  Primals<HMCWrapper::Field> LightStrangeLinks = Policy.promise();
  Primals<HMCWrapper::Field> CharmLinks = Policy.promise(CharmNaikEpsilon);

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
#ifdef GRID_MILC_RNG_AUDIT
  MilcAudit::refreshedFields = {&LightStrangePF.Phi,&RegulatorPF1.Phi,
    &RegulatorPF2.Phi,&RegulatorPF3.Phi,&CharmPF.Phi};
#endif

  // Eq. A2, beta = 10/g^2. Tadpole improvement is gauge-only.
  // Count physical sea-quark flavors; the regulator determinants cancel.
  const int nf = LightParams.nf + StrangeParams.nf + CharmParams.nf;
  RealD u0 = Action.TadpoleFactor;
  PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl> GaugeAction(
    GridPtr, 
    Action.BareGaugeCoupling, 
    1.0, 
    oneLoopMILCRectangleCoefficient(u0, nf), 
    oneLoopMILCParallelogramCoefficient(u0, nf)
  );

  LightStrangePF.signContract(LightStrangeLinks, LightStrangeLinks, LightStrangeLinks);
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
