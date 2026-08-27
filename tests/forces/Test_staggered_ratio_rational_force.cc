/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/forces/Test_staggered_ratio_rational_force.cc

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

using namespace std;
using namespace Grid;

typedef NaiveStaggeredFermionD::FermionField FermionField;
typedef SchurStaggeredOperator<FermionOperator<StaggeredImplD>, FermionField> SchurOp;

// Largest eigenvalue of A = m^2 - Meooe Meooe on the even checkerboard
// by power iteration
RealD LargestEigenvalue(
  FermionOperator<StaggeredImplD>& Ds,
  GridCartesian* UGrid,
  GridRedBlackCartesian* RBGrid,
  GridParallelRNG& pRNG,
  int iters
) {
  SchurOp A(Ds);
  FermionField vfull(UGrid);
  FermionField v(RBGrid);
  FermionField w(RBGrid);

  gaussian(pRNG, vfull);
  pickCheckerboard(Even, v, vfull);
  w.Checkerboard() = Even;

  RealD lambda = 0.0;
  for (int i = 0; i < iters; ++i) {
    A.HermOp(v, w);
    lambda = innerProduct(v, w).real() / norm2(v);
    v = w * (1.0 / std::sqrt(norm2(w)));
  }
  return lambda;
}

// Finite-difference force check
std::pair<RealD, RealD> ForceCheck(
  const std::string& name,
  Action<LatticeGaugeField>& PF,
  LatticeGaugeField& U,
  LatticeGaugeField& Uprime,
  LatticeGaugeField& mom,
  GridCartesian* UGrid,
  RealD dt
) {
  std::cout << GridLogMessage << "==== " << name << " ====" << std::endl;

  RealD S = PF.S(U);

  LatticeGaugeField UdSdU(UGrid);
  PF.deriv(U, UdSdU);
  UdSdU = Ta(UdSdU);

  RealD Sprime = PF.S(Uprime);

  LatticeColourMatrix mommu(UGrid);
  LatticeColourMatrix forcemu(UGrid);

  for (int mu = 0; mu < Nd; mu++) {
    mommu = PeekIndex<LorentzIndex>(UdSdU, mu);
    mommu = Ta(mommu) * 2.0;
    PokeIndex<LorentzIndex>(UdSdU, mommu, mu);
  }

  LatticeComplex dS(UGrid);
  dS = Zero();
  for (int mu = 0; mu < Nd; mu++) {
    forcemu = PeekIndex<LorentzIndex>(UdSdU, mu);
    mommu = PeekIndex<LorentzIndex>(mom, mu);
    dS = dS + trace(mommu * forcemu) * dt;
  }

  ComplexD dSpredC = sum(dS);
  RealD dSmeas = Sprime - S;
  RealD dSpred = real(dSpredC);

  std::cout << GridLogMessage << name << " S           " << S << std::endl;
  std::cout << GridLogMessage << name << " Sprime      " << Sprime << std::endl;
  std::cout << GridLogMessage << name << " dS          " << dSmeas << std::endl;
  std::cout << GridLogMessage << name << " predict dS  " << dSpred << std::endl;
  std::cout << GridLogMessage << name << " ratio       "
            << dSmeas / dSpred << std::endl;

  return {dSmeas, dSpred};
}

void banner(const std::string& s) {
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "########################################" << std::endl;
  std::cout << GridLogMessage << "# " << s << std::endl;
  std::cout << GridLogMessage << "########################################" << std::endl;
}

int main(int argc, char** argv)
{
  Grid_init(&argc, &argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian           UGrid(latt_size, simd_layout, mpi_layout);
  GridRedBlackCartesian  RBGrid(&UGrid);

  std::cout << GridLogMessage << "Grid is setup to use "
            << GridThread::GetThreads() << " threads" << std::endl;

  GridParallelRNG pRNG(&UGrid);
  GridSerialRNG   sRNG;
  pRNG.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));
  sRNG.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));

  LatticeGaugeField U(&UGrid);
  SU<Nc>::ColdConfiguration(pRNG, U);

  RealD c1 = 1.0;
  RealD u0 = 1.0;

  ////////////////////////////////////////////////////////////////////
  // Two genuinely non-commuting kernels. c1 and u0 enter only as the
  // overall scalar 0.5*c1/u0 on the links, so naive operators differing
  // in mass, c1 or u0 have proportional kernels and commute. Boundary
  // phases do not: they are applied on the final coordinate slice during
  // the gauge import, so these two operators do not commute.
  ////////////////////////////////////////////////////////////////////
  StaggeredImplParams paramsNum(std::vector<Complex>({1.0, 1.0, 1.0,  1.0})); // periodic
  StaggeredImplParams paramsDen(std::vector<Complex>({1.0, 1.0, 1.0, -1.0})); // antiperiodic in time

  RealD massNum = 0.3;
  RealD massDen = 0.1;

  NaiveStaggeredFermionD DsNum(U, UGrid, RBGrid, massNum, c1, u0, paramsNum);
  NaiveStaggeredFermionD DsDen(U, UGrid, RBGrid, massDen, c1, u0, paramsDen);

  // Equal-operator pair for test 1: same mass AND same boundary phases
  NaiveStaggeredFermionD DsSame(U, UGrid, RBGrid, massNum, c1, u0, paramsNum);

  ////////////////////////////////////////////////////////////////////
  // Separate Remez bounds per operator: different kernels, different spectra
  ////////////////////////////////////////////////////////////////////
  RealD hiNum = 1.5 * LargestEigenvalue(DsNum, &UGrid, &RBGrid, pRNG, 200);
  RealD hiDen = 1.5 * LargestEigenvalue(DsDen, &UGrid, &RBGrid, pRNG, 200);
  RealD loNum = 0.5 * massNum * massNum;
  RealD loDen = 0.5 * massDen * massDen;

  std::cout << GridLogMessage << "Remez bounds num [" << loNum << "," << hiNum << "]" << std::endl;
  std::cout << GridLogMessage << "Remez bounds den [" << loDen << "," << hiDen << "]" << std::endl;

  int   degree    = 16;
  RealD tolerance = 1.0e-12;
  int   precision = 64;
  int   maxIter   = 20000;

  ConjugateGradient<FermionField> CG(1.0e-12, 20000);

  auto mkParams = [&](int nf, RealD lo, RealD hi) {
    return StaggeredRationalActionParams(
      nf, lo, hi, maxIter, tolerance, degree, tolerance, degree, precision, 0);
  };

  ////////////////////////////////////////////////////////////////////
  // Force scale, for judging what "zero" means in test 1
  ////////////////////////////////////////////////////////////////////
  FourFlavorStaggeredEvenEvenPseudoFermionAction<StaggeredImplD> PF4(DsNum, CG, CG);
  PF4.refresh(U, sRNG, pRNG);
  LatticeGaugeField F4(&UGrid);
  PF4.deriv(U, F4);
  RealD scaleF = std::sqrt(norm2(F4));
  std::cout << GridLogMessage << "four-flavor force scale |F| = " << scaleF << std::endl;

  int nf = 2;

  ////////////////////////////////////////////////////////////////////
  // (1) Equal operators. A_M = A_L, so
  //       S = Phi^dag A^(Nf/8) A^(-Nf/4) A^(Nf/8) Phi = |Phi|^2
  //     and dS/dU = 0 identically. The two force sums must cancel term
  //     by term, which is not visible by inspection.
  ////////////////////////////////////////////////////////////////////
  banner("(1) equal operators: S = |Phi|^2, dSdU = 0");

  StaggeredEvenEvenRatioRational<StaggeredImplD>
    PFsame(DsNum, DsSame, mkParams(nf, loNum, hiNum), mkParams(nf, loNum, hiNum));
  PFsame.refresh(U, sRNG, pRNG);

  RealD S_same    = PFsame.S(U);
  RealD S_same_ex = norm2(PFsame.Phi);
  LatticeGaugeField F_same(&UGrid);
  PFsame.deriv(U, F_same);

  RealD relS_same = std::fabs(S_same - S_same_ex) / std::fabs(S_same_ex);
  RealD relF_same = std::sqrt(norm2(F_same)) / scaleF;

  std::cout << GridLogMessage << "S (class) " << S_same
            << "   |Phi|^2 " << S_same_ex
            << "   relative " << relS_same << std::endl;
  std::cout << GridLogMessage << "|F| " << std::sqrt(norm2(F_same))
            << "   relative to four-flavor force scale " << relF_same << std::endl;

  ////////////////////////////////////////////////////////////////////
  // (2) Split operators, action reconstructed from the four-flavor class
  //     evaluated at the effective masses sqrt(m_den^2 + t_k).
  //
  //       X = R_h(A_M) Phi
  //       S = <X, R_n(A_L) X>
  //         = norm |X|^2 + sum_k s_k X^dag (A_L + t_k)^-1 X
  //
  //     The pole terms are exactly the four-flavor even-even action of the
  //     denominator operator at the effective mass, driven from X rather
  //     than Phi. Tests the identity A(m) + t = A(sqrt(m^2 + t)) on the
  //     denominator, and that the two approximations chain correctly.
  ////////////////////////////////////////////////////////////////////
  banner("(2) split operators: action reconstruction from four-flavor class");

  StaggeredEvenEvenRatioRational<StaggeredImplD>
    PFsplit(DsNum, DsDen, mkParams(nf, loNum, hiNum), mkParams(nf, loDen, hiDen));
  PFsplit.refresh(U, sRNG, pRNG);

  RealD S_cls = PFsplit.S(U);

  // Rebuild the class's own two approximations, deterministically
  AlgRemez remezNum(loNum, hiNum, precision);
  remezNum.generateApprox(degree, nf, 8);
  MultiShiftFunction Rh;
  Rh.Init(remezNum, tolerance, false);          // x^(+nf/8) on A_M

  AlgRemez remezDen(loDen, hiDen, precision);
  remezDen.generateApprox(degree, nf, 4);
  MultiShiftFunction Rn;
  Rn.Init(remezDen, tolerance, true);           // x^(-nf/4) on A_L

  // X = R_h(A_M) Phi
  SchurOp AM(DsNum);
  ConjugateGradientMultiShift<FermionField> msNum(maxIter, Rh);
  std::vector<FermionField> Xs(Rh.poles.size(), &RBGrid);
  FermionField X(&RBGrid);
  for (auto& x : Xs) { x.Checkerboard() = Even; }
  X.Checkerboard() = Even;
  msNum(AM, PFsplit.Phi, Xs, X);

  RealD S_ref = Rn.norm * norm2(X);
  for (int k = 0; k < (int)Rn.poles.size(); ++k) {
    RealD m_eff2 = massDen * massDen + Rn.poles[k];
    GRID_ASSERT(m_eff2 > 0.0 && "effective mass squared must be positive");
    RealD m_eff = std::sqrt(m_eff2);

    // the denominator operator's boundary phases, at the effective mass
    NaiveStaggeredFermionD Dk(U, UGrid, RBGrid, m_eff, c1, u0, paramsDen);
    FourFlavorStaggeredEvenEvenPseudoFermionAction<StaggeredImplD> PFk(Dk, CG, CG);
    PFk.Phi = X;
    S_ref += Rn.residues[k] * PFk.S(U);
  }

  RealD relS_split = std::fabs(S_cls - S_ref) / std::fabs(S_ref);
  std::cout << GridLogMessage << "S (class) " << S_cls
            << "   S (four-flavor sum) " << S_ref
            << "   relative " << relS_split << std::endl;

  ////////////////////////////////////////////////////////////////////
  // (3) Finite-difference derivative check on the split operators
  ////////////////////////////////////////////////////////////////////
  banner("(3) finite-difference derivative check, split operators");

  RealD dt = 1.0e-5;

  LatticeColourMatrix mommu(&UGrid);
  LatticeGaugeField   mom(&UGrid);
  LatticeGaugeField   Uprime(&UGrid);

  for (int mu = 0; mu < Nd; mu++) {
    SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, mommu);
    PokeIndex<LorentzIndex>(mom, mommu, mu);

    autoView(U_v, U, CpuRead);
    autoView(mom_v, mom, CpuRead);
    autoView(Uprime_v, Uprime, CpuWrite);
    thread_foreach(i, mom_v, {
      Uprime_v[i](mu) = U_v[i](mu);
      Uprime_v[i](mu) += mom_v[i](mu) * U_v[i](mu) * dt;
      Uprime_v[i](mu) += mom_v[i](mu) * mom_v[i](mu) * U_v[i](mu) * (dt * dt / 2.0);
      Uprime_v[i](mu) += mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * U_v[i](mu) * (dt * dt * dt / 6.0);
      Uprime_v[i](mu) += mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * U_v[i](mu) * (dt * dt * dt * dt / 24.0);
      Uprime_v[i](mu) += mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * U_v[i](mu) * (dt * dt * dt * dt * dt / 120.0);
      Uprime_v[i](mu) += mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * mom_v[i](mu) * U_v[i](mu) * (dt * dt * dt * dt * dt * dt / 720.0);
    });
  }

  // Calibrate the sign convention against the known-good four-flavor class
  auto ref = ForceCheck("four-flavor ", PF4, U, Uprime, mom, &UGrid, dt);
  RealD refRatio = ref.first / ref.second;
  std::cout << GridLogMessage
            << "reference dS/predict = " << refRatio
            << "  (this is the value a correct force reproduces)" << std::endl;

  std::vector<std::pair<int, RealD>> results;
  for (int nfi : {1, 2, 3}) {
    StaggeredEvenEvenRatioRational<StaggeredImplD>
      PF(DsNum, DsDen, mkParams(nfi, loNum, hiNum), mkParams(nfi, loDen, hiDen));
    PF.refresh(U, sRNG, pRNG);

    std::ostringstream nm;
    nm << "ratio rational Nf=" << nfi << " ";
    auto r = ForceCheck(nm.str(), PF, U, Uprime, mom, &UGrid, dt);
    results.push_back({nfi, r.first / r.second});
  }

  ////////////////////////////////////////////////////////////////////
  // Summary
  ////////////////////////////////////////////////////////////////////
  banner("summary");
  std::cout << GridLogMessage << "(1) equal operators   relative dS = " << relS_same
            << " , |F|/|F_4f| = " << relF_same << std::endl;
  std::cout << GridLogMessage << "(2) reconstruction    relative dS = " << relS_split << std::endl;
  std::cout << GridLogMessage
            << "pass condition: dS/predict = -1 (this harness reports predict dS"
            << " with the opposite sign to the measured dS)" << std::endl;
  std::cout << GridLogMessage << "(3) four-flavor       dS/predict = " << refRatio
            << " , |ratio+1| = " << std::fabs(refRatio + 1.0) << std::endl;
  for (auto& r : results) {
    std::cout << GridLogMessage << "(3) Nf=" << r.first
              << "              dS/predict = " << r.second
              << " , |ratio+1| = " << std::fabs(r.second + 1.0) << std::endl;
  }

  GRID_ASSERT(relS_same < 1.0e-8 && "equal operators must give S = |Phi|^2");
  GRID_ASSERT(relF_same < 1.0e-6 && "equal operators must give a vanishing force");
  GRID_ASSERT(relS_split < 1.0e-8 && "action must match four-flavor reconstruction");
  GRID_ASSERT(std::fabs(refRatio + 1.0) < 0.1 &&
              "four-flavor reference itself failed the finite-difference check");
  for (auto& r : results) {
    GRID_ASSERT(std::fabs(r.second + 1.0) < 0.1 &&
                "ratio rational force disagrees with finite difference");
  }

  std::cout << GridLogMessage << "Done" << std::endl;
  Grid_finalize();
}
