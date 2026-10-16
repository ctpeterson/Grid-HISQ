/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./HMC/MobiusManyFlavourStaggeredPauliVillars.cc

Copyright (C) 2015-2016

Author: Peter Boyle <pabobyle@ph.ed.ac.uk>
Author: Guido Cossu <guido.cossu@ed.ac.uk>
Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com> (Pauli-Villars)

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
#include "disable_examples_without_instantiations.h"
#ifdef ENABLE_FERMION_INSTANTIATIONS

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);
  struct FermionParameters: Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(FermionParameters,
                        int, Ls,
                        double, mass,
                        double, M5,
                        double, b,
                        double, c,
                        double, StoppingCondition,
                        int, MaxCGIterations,
                        bool, ApplySmearing
    );
  };

  struct PauliVillarsParameters: Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(PauliVillarsParameters,
                        double, mass,
                        double, StoppingCondition,
                        int, MaxCGIterations,
                        bool, ApplySmearing
    );
  };

  struct MobiusHMCParameters: Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(MobiusHMCParameters,
                        double, gauge_beta,
                        FermionParameters, Mobius,
                        PauliVillarsParameters, PauliVillars,
    );
    template <class ReaderClass >
    MobiusHMCParameters(Reader<ReaderClass>& Reader){read(Reader, "Action", *this);}
  };

  struct StoutSmearingParameters: Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(StoutSmearingParameters,
                        double, rho,
                        Integer, Nsmear
    );
    template <class ReaderClass >
    StoutSmearingParameters(Reader<ReaderClass>& Reader){read(Reader, "StoutSmearing", *this);}
  };
NAMESPACE_END(Grid);

int main(int argc, char **argv){
  using namespace Grid;

  Grid_init(&argc, &argv);
  {
    int threads = GridThread::GetThreads();
    std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;

    // Typedefs to simplify notation
    typedef GenericHMCRunnerD<MinimumNorm2> HMCWrapper;  // Uses the default minimum norm
    typedef WilsonImplD FermionImplPolicy;
    typedef MobiusFermionD FermionAction;
    typedef NaiveStaggeredFermionD PauliVillarsAction;
    typedef typename FermionAction::FermionField FermionField;
    typedef typename PauliVillarsAction::FermionField PauliVillarsField;
    typedef Grid::XmlReader Serialiser;
    typedef StaggeredImplD StagImplPolicy;

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    HMCWrapper TheHMC;
    TheHMC.ReadCommandLine(argc, argv); // these can be parameters from file

    // Reader, file should come from command line
    if (TheHMC.ParameterFile.empty()){
      std::cout << "Input file not specified."
                << "Use --ParameterFile option in the command line.\nAborting"
                << std::endl;
      Grid_finalize();
      return 1;
    }
    Serialiser Reader(TheHMC.ParameterFile);
    MobiusHMCParameters MyParams(Reader);

    // Apply smearing to the fermionic action
    bool ApplySmearing = MyParams.Mobius.ApplySmearing;
    bool ApplyPVSmearing = MyParams.PauliVillars.ApplySmearing;

    // Grid from the command line
    TheHMC.Resources.AddFourDimGrid("gauge");

    // Checkpointer definition (Name: Checkpointer)
    CheckpointerParameters CPparams(Reader);

    TheHMC.Resources.LoadNerscCheckpointer(CPparams);

    // RNG definition (Name: RandomNumberGenerator)
    RNGModuleParameters RNGpar(Reader);
    TheHMC.Resources.SetRNGSeeds(RNGpar);

    // Construct observables
    typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
    TheHMC.Resources.AddObservable<PlaqObs>();

    /////////////////////////////////////////////////////////////

    SymanzikGaugeActionD Syzaction(MyParams.gauge_beta);

    const int Ls = MyParams.Mobius.Ls;
    auto GridPtr   = TheHMC.Resources.GetCartesian();
    auto GridRBPtr = TheHMC.Resources.GetRBCartesian();
    std::unique_ptr<GridCartesian> FGrid(SpaceTimeGrid::makeFiveDimGrid(Ls,GridPtr));
    std::unique_ptr<GridRedBlackCartesian> FrbGrid(SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,GridPtr));

    // temporarily need a gauge field
    LatticeGaugeFieldD U(GridPtr);
    SU<Nc>::ColdConfiguration(U);

    RealD mass = MyParams.Mobius.mass;
    RealD pv   = 1.0; // Standard DWF regulator in the determinant ratio.
    RealD M5   = MyParams.Mobius.M5;
    // Note: IroIro and Grid notation for b and c differ
    RealD b   = MyParams.Mobius.b;
    RealD c   = MyParams.Mobius.c;
    RealD mpv = MyParams.PauliVillars.mass;

    // Periodic spatial and anti-periodic temporal fermion boundary conditions.
    std::vector<Complex> boundary = {1,1,1,-1};
    FermionAction::ImplParams Params(boundary);
    PauliVillarsAction::ImplParams PVParams(boundary);

    ConjugateGradient<FermionField> CG(MyParams.Mobius.StoppingCondition,MyParams.Mobius.MaxCGIterations);
    ConjugateGradient<PauliVillarsField> PVCG(
      MyParams.PauliVillars.StoppingCondition,
      MyParams.PauliVillars.MaxCGIterations
    );

    RealD u0      = 1.0;
    RealD c1      = 1.0;

    FermionAction NumOp(U,*FGrid,*FrbGrid,*GridPtr,*GridRBPtr,pv,M5,b,c,Params);
    FermionAction DenOp(U,*FGrid,*FrbGrid,*GridPtr,*GridRBPtr,mass,M5,b,c,Params);
    PauliVillarsAction StagPV(U,*GridPtr,*GridRBPtr,mpv,c1,u0,PVParams);

    // Original field content: two two-flavour Mobius ratios (four DWF flavours).
    TwoFlavourEvenOddRatioPseudoFermionAction<FermionImplPolicy> Nf2a(NumOp,DenOp,CG,CG);
    TwoFlavourEvenOddRatioPseudoFermionAction<FermionImplPolicy> Nf2b(NumOp,DenOp,CG,CG);

    // Eight additional four-flavor staggered PV fields, separate from the
    // standard DWF regulator above.

    FourFlavorStaggeredPauliVillarsPseudoFermionAction<StagImplPolicy> Boson1(StagPV,PVCG);
    FourFlavorStaggeredPauliVillarsPseudoFermionAction<StagImplPolicy> Boson2(StagPV,PVCG);
    FourFlavorStaggeredPauliVillarsPseudoFermionAction<StagImplPolicy> Boson3(StagPV,PVCG);
    FourFlavorStaggeredPauliVillarsPseudoFermionAction<StagImplPolicy> Boson4(StagPV,PVCG);

    FourFlavorStaggeredPauliVillarsPseudoFermionAction<StagImplPolicy> Boson5(StagPV,PVCG);
    FourFlavorStaggeredPauliVillarsPseudoFermionAction<StagImplPolicy> Boson6(StagPV,PVCG);
    FourFlavorStaggeredPauliVillarsPseudoFermionAction<StagImplPolicy> Boson7(StagPV,PVCG);
    FourFlavorStaggeredPauliVillarsPseudoFermionAction<StagImplPolicy> Boson8(StagPV,PVCG);

    // Legacy path: select smeared links with is_smeared; no link contracts.
    Nf2a.is_smeared = ApplySmearing;
    Nf2b.is_smeared = ApplySmearing;

    Boson1.is_smeared = ApplyPVSmearing;
    Boson2.is_smeared = ApplyPVSmearing;
    Boson3.is_smeared = ApplyPVSmearing;
    Boson4.is_smeared = ApplyPVSmearing;

    Boson5.is_smeared = ApplyPVSmearing;
    Boson6.is_smeared = ApplyPVSmearing;
    Boson7.is_smeared = ApplyPVSmearing;
    Boson8.is_smeared = ApplyPVSmearing;

    ActionLevel<HMCWrapper::Field> Level1(1);

    Level1.push_back(&Nf2a);
    Level1.push_back(&Nf2b);

    Level1.push_back(&Boson1);
    Level1.push_back(&Boson2);
    Level1.push_back(&Boson3);
    Level1.push_back(&Boson4);

    Level1.push_back(&Boson5);
    Level1.push_back(&Boson6);
    Level1.push_back(&Boson7);
    Level1.push_back(&Boson8);

    ActionLevel<HMCWrapper::Field> Level2(4);

    Level2.push_back(&Syzaction);

    TheHMC.TheAction.push_back(Level1);
    TheHMC.TheAction.push_back(Level2);

    /////////////////////////////////////////////////////////////
    // HMC parameters are serialisable
    TheHMC.Parameters.initialize(Reader);
    // Command-line run/restart controls override the parameter file.
    TheHMC.ReadCommandLine(argc, argv);

    // Run with the legacy stout configuration when either sector uses it.
    if (ApplySmearing || ApplyPVSmearing){
      StoutSmearingParameters SmPar(Reader);
      GRID_ASSERT(SmPar.Nsmear >= 0);
      Smear_Stout<HMCWrapper::ImplPolicy> Stout(SmPar.rho);
      StoutConfiguration<HMCWrapper::ImplPolicy> SmearingPolicy(GridPtr, SmPar.Nsmear, Stout);
      TheHMC.Run(SmearingPolicy); // for smearing
    }
    else {TheHMC.Run();}
  }
  Grid_finalize();
}

#endif // ENABLE_FERMION_INSTANTIATIONS
