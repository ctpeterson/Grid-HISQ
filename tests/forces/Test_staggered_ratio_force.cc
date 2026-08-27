/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/forces/Test_staggered_ratio_force.cc

Copyright (C) 2015

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
#include <Grid/Grid.h>

using namespace Grid;

int main(int argc, char** argv)
{
  Grid_init(&argc, &argv);

  typedef NaiveStaggeredFermionD::FermionField FermionField;

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian           Grid(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian RBGrid(&Grid);

  std::cout << GridLogMessage << "Threads: " << GridThread::GetThreads() << std::endl;

  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG pRNG(&Grid);  pRNG.SeedFixedIntegers(seeds);
  GridSerialRNG   sRNG;         sRNG.SeedFixedIntegers(seeds);

  LatticeGaugeField U(&Grid);
  SU<Nc>::HotConfiguration(pRNG, U);

  ////////////////////////////////////////////////////////////////////////
  // Numerator and denominator must share links, c1 and u0 and differ only
  // in mass: the Mpc_num - Mpc_den = Delta collapse in _deriv requires the
  // massless Dirac operators to commute.
  ////////////////////////////////////////////////////////////////////////
  RealD massNum = 0.5;
  RealD massDen = 0.01;
  RealD c1      = 1.0;
  RealD u0      = 1.0;

  NaiveStaggeredFermionD NumOp(U, Grid, RBGrid, massNum, c1, u0);
  NaiveStaggeredFermionD DenOp(U, Grid, RBGrid, massDen, c1, u0);

  // Tight tolerance: at production tolerance the two force branches inside
  // _deriv differ by solver noise at about the level a sign error would show.
  ConjugateGradient<FermionField> CGderiv (1.0e-12, 30000);
  ConjugateGradient<FermionField> CGaction(1.0e-12, 30000);

  // ctor is (NumOp, DenOp, DerivativeSolver, ActionSolver)
  FourFlavorStaggeredEvenEvenRatioPseudoFermionAction<StaggeredImplD>
    PF(NumOp, DenOp, CGderiv, CGaction);

  std::cout << GridLogMessage << "massNum = " << massNum
            << "   massDen = "  << massDen
            << "   Delta = "    << massNum*massNum - massDen*massDen
            << std::endl;

  PF.refresh(U, sRNG, pRNG);
  std::cout << GridLogMessage << "ITERS refresh (ActionSolver, NumOp MdagM full grid) = "
            << CGaction.IterationsToComplete << std::endl;

  RealD S = PF.S(U);
  std::cout << GridLogMessage << "ITERS action  (ActionSolver, DenOp MdagM full grid) = "
            << CGaction.IterationsToComplete << std::endl;
  std::cout << GridLogMessage << "S = " << S << std::endl;

  LatticeGaugeField dSdU(&Grid);
  dSdU = Zero();

  // _deriv computes both the full-grid MdagM force and the Mpc-form force and
  // reports the norm2 of their difference.
  /*
  PF.deriv(U, dSdU);

  // DerivativeSolver runs twice inside _deriv: first the HEAD branch (full-grid
  // MdagM), then the Mpc branch. IterationsToComplete holds the LAST one, so
  // this number is unambiguously the Mpc solve.
  std::cout << GridLogMessage << "ITERS deriv   (DerivativeSolver, LAST solve = Mpc even cb) = "
            << CGderiv.IterationsToComplete << std::endl;

  RealD n2 = norm2(dSdU);
  std::cout << GridLogMessage << "norm2(dSdU)     = " << n2                << std::endl;
  std::cout << GridLogMessage << "norm2(Ta(dSdU)) = " << norm2(Ta(dSdU))   << std::endl;
  std::cout << GridLogMessage << "Compare the DIFFERENCE above against norm2(dSdU): "
            << "agreement means DIFFERENCE/norm2(dSdU) ~ 1e-22, "
            << "a flipped overall sign gives exactly 4."
            << std::endl;
  */

  Grid_finalize();
}
