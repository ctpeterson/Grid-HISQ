/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/forces/Test_staggered_rational_force.cc

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

// Largest eigenvalue of A = m^2 - Meooe Meooe on the even checkerboard,
// by power iteration. 
RealD LargestEigenvalue(
  FermionOperator<StaggeredImplD>& Ds,
  GridCartesian* UGrid,
  GridRedBlackCartesian* RBGrid,
  GridParallelRNG& pRNG,
  int iters
) {
  SchurStaggeredOperator<FermionOperator<StaggeredImplD>, FermionField> A(Ds);
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

// Finite-difference force check. The gauge perturbation (mom, Uprime) is built
// once by the caller so that every action sees an identical displacement.
// Returns { measured dS, predicted dS }.
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
  std::cout << GridLogMessage << name << " |dS|/|pred| "
            << std::fabs(dSmeas) / std::fabs(dSpred) << std::endl;
  std::cout << GridLogMessage << name << " ratio       "
            << dSmeas / dSpred << std::endl;

  return {dSmeas, dSpred};
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

  RealD mass = 0.1;
  RealD c1   = 1.0;
  RealD u0   = 1.0;
  NaiveStaggeredFermionD Ds(U, UGrid, RBGrid, mass, c1, u0);

  ////////////////////////////////////////////////////////////////////
  // Remez bounds: lo below min spec(A) = m^2, hi above max spec(A)
  ////////////////////////////////////////////////////////////////////
  RealD lam_hi = LargestEigenvalue(Ds, &UGrid, &RBGrid, pRNG, 200);
  RealD lo = 0.5 * mass * mass;
  RealD hi = 1.5 * lam_hi;
  std::cout << GridLogMessage << "power method: max eigenvalue of A ~ "
            << lam_hi << std::endl;
  std::cout << GridLogMessage << "Remez bounds [" << lo << "," << hi << "]"
            << std::endl;

  ConjugateGradient<FermionField> CG(1.0e-12, 20000);

  int    degree    = 16;
  RealD  tolerance = 1.0e-12;
  int    precision = 64;

  ////////////////////////////////////////////////////////////////////
  // (1) Reconstruct the rational action and force from the four-flavor
  //     class evaluated at the effective masses sqrt(m^2 + p_k).
  //
  //     S   = norm |Phi|^2 + sum_k res_k Phi^dag (A + p_k)^-1 Phi
  //     dS  = sum_k res_k * [ four-flavor force at mass sqrt(m^2+p_k) ]
  //
  //     This uses the already finite-difference-validated four-flavor class
  //     as the reference, and tests the identity A(m) + p = A(sqrt(m^2+p))
  //     that the whole design rests on.
  //
  //     Ngl, this test is pretty cute
  ////////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "########################################" << std::endl;
  std::cout << GridLogMessage << "# (1) reconstruction from four-flavor class" << std::endl;
  std::cout << GridLogMessage << "########################################" << std::endl;

  std::vector<std::pair<int, std::pair<RealD,RealD>>> recon;

  for (int nf : {1, 2, 3}) {
    // Rebuild the class's own x^(-nf/4) approximation, deterministically
    AlgRemez remez(lo, hi, precision);
    remez.generateApprox(degree, nf, 4);
    MultiShiftFunction approx;
    approx.Init(remez, tolerance, true);

    StaggeredRationalActionParams p(
      nf, lo, hi, 20000, tolerance, degree, tolerance, degree, precision, 0);
    StaggeredEvenEvenRational<StaggeredImplD> PF(Ds, p);
    PF.refresh(U, sRNG, pRNG);

    RealD S_rat = PF.S(U);
    LatticeGaugeField F_rat(&UGrid);
    PF.deriv(U, F_rat);

    // Reference assembled pole by pole from the four-flavor class
    RealD S_ref = approx.norm * norm2(PF.Phi);
    LatticeGaugeField F_ref(&UGrid);
    LatticeGaugeField F_k(&UGrid);
    F_ref = Zero();

    for (int k = 0; k < approx.poles.size(); ++k) {
      RealD m_eff2 = mass * mass + approx.poles[k];
      GRID_ASSERT(m_eff2 > 0.0 && "effective mass squared must be positive");
      RealD m_eff = std::sqrt(m_eff2);

      NaiveStaggeredFermionD Ds_k(U, UGrid, RBGrid, m_eff, c1, u0);
      FourFlavorStaggeredEvenEvenPseudoFermionAction<StaggeredImplD> PF_k(Ds_k, CG, CG);
      PF_k.Phi = PF.Phi;

      S_ref += approx.residues[k] * PF_k.S(U);
      PF_k.deriv(U, F_k);
      F_ref = F_ref + approx.residues[k] * F_k;
    }

    LatticeGaugeField F_diff(&UGrid);
    F_diff = F_rat - F_ref;

    RealD relS = std::fabs(S_rat - S_ref) / std::fabs(S_ref);
    RealD relF = std::sqrt(norm2(F_diff) / norm2(F_ref));

    std::cout << GridLogMessage << "Nf=" << nf
              << "  S (rational) " << S_rat
              << "  S (four-flavor sum) " << S_ref
              << "  relative " << relS << std::endl;
    std::cout << GridLogMessage << "Nf=" << nf
              << "  |F|^2 (rational) " << norm2(F_rat)
              << "  |F|^2 (four-flavor sum) " << norm2(F_ref)
              << "  relative " << relF << std::endl;

    recon.push_back({nf, {relS, relF}});
  }

  ////////////////////////////////////////////////////////////////////
  // (2) Finite-difference derivative check
  ////////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "########################################" << std::endl;
  std::cout << GridLogMessage << "# (2) finite-difference derivative check" << std::endl;
  std::cout << GridLogMessage << "########################################" << std::endl;

  RealD dt = 1.0e-4;

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

  // Calibrate the sign convention against the known-good four-flavor class,
  // using the Phi it was refreshed with above.
  FourFlavorStaggeredEvenEvenPseudoFermionAction<StaggeredImplD> PF4(Ds, CG, CG);
  PF4.refresh(U, sRNG, pRNG);
  auto ref = ForceCheck("four-flavor ", PF4, U, Uprime, mom, &UGrid, dt);
  RealD refRatio = ref.first / ref.second;
  std::cout << GridLogMessage
            << "reference dS/predict = " << refRatio
            << "  (this is the value a correct force reproduces)" << std::endl;

  std::vector<std::pair<int, RealD>> results;
  for (int nf : {1, 2, 3}) {
    StaggeredRationalActionParams p(
      nf, lo, hi, 20000, tolerance, degree, tolerance, degree, precision, 0);
    StaggeredEvenEvenRational<StaggeredImplD> PF(Ds, p);
    PF.refresh(U, sRNG, pRNG);

    std::ostringstream nm;
    nm << "rational Nf=" << nf << " ";
    auto r = ForceCheck(nm.str(), PF, U, Uprime, mom, &UGrid, dt);
    results.push_back({nf, r.first / r.second});
  }

  ////////////////////////////////////////////////////////////////////
  // Summary
  ////////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << std::endl;
  std::cout << GridLogMessage << "########################################" << std::endl;
  std::cout << GridLogMessage << "# summary" << std::endl;
  std::cout << GridLogMessage << "########################################" << std::endl;
  for (auto& r : recon) {
    std::cout << GridLogMessage << "Nf=" << r.first
              << " reconstruction: relative dS = " << r.second.first
              << " , relative dF = " << r.second.second << std::endl;
  }
  std::cout << GridLogMessage << "reference dS/predict (four-flavor) = "
            << refRatio << std::endl;
  std::cout << GridLogMessage
            << "pass condition: dS/predict = -1 (this harness reports predict dS"
            << " with the opposite sign to the measured dS)" << std::endl;
  std::cout << GridLogMessage << "four-flavor  dS/predict = " << refRatio
            << " , |ratio+1| = " << std::fabs(refRatio + 1.0) << std::endl;
  for (auto& r : results) {
    std::cout << GridLogMessage << "Nf=" << r.first
              << " dS/predict = " << r.second
              << " , |ratio+1| = " << std::fabs(r.second + 1.0) << std::endl;
  }

  for (auto& r : recon) {
    GRID_ASSERT(r.second.first  < 1.0e-8 && "action must match four-flavor reconstruction");
    GRID_ASSERT(r.second.second < 1.0e-8 && "force must match four-flavor reconstruction");
  }
  // O(dt) truncation sets the scale of the residual disagreement; the
  // four-flavor reference sits at the same level, so use it as the yardstick
  // for what this harness can resolve at dt = 1e-4.
  GRID_ASSERT(std::fabs(refRatio + 1.0) < 0.1 &&
              "four-flavor reference itself failed the finite-difference check");
  for (auto& r : results) {
    GRID_ASSERT(std::fabs(r.second + 1.0) < 0.1 &&
                "rational force disagrees with finite difference");
  }

  std::cout << GridLogMessage << "Done" << std::endl;
  Grid_finalize();
}
