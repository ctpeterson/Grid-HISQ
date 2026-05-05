/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/smearing/Test_unitary_projection.cc

Copyright (C) 2023

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

// Note the Claude Code was used to generate and iterate on this test file

#include <Grid/Grid.h>
#include <Grid/qcd/utils/UnitaryProjection.h>

using namespace Grid;

int main (int argc, char ** argv) {
  Grid_init(&argc,&argv);

  /* setup */

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();

  GridCartesian         Grid(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian RBGrid(&Grid);

  int threads = GridThread::GetThreads();
  std::cout<<GridLogMessage << "Grid is setup to use "<<threads<<" threads"<<std::endl;

  /* create unitary projection object */

  UnitaryProjectionContext ctx(CayleyHamiltonProjection, MIMDCollaborationDerivative);
  
  // arXiv:1004.0342
  ctx.setDerivativeEigenvalueCutoff(5e-5);
  ctx.setBackupSVD(true); // for fun: turn off and watch tests fail
  ctx.setRelativeSVDTolerance(1e-8);
  ctx.setAbsoluteSVDTolerance(1e-8);

  UnitaryProjection<PeriodicGimplD> proj(ctx);

  /* stress test unitary projection */

  // fill every site of a link field with the same scalar matrix
  auto fillField = [](LatticeColourMatrixD& field, const ColourMatrixD& m) {
    GridBase* grid = field.Grid();
    autoView(fld_v, field, CpuWrite);
    thread_for(n, grid->lSites(), {
      Coordinate lcoor;
      grid->LocalIndexToLocalCoor(n, lcoor);
      pokeLocalSite(m, fld_v, lcoor);
    });
  };

  // per-site relative unitarity residual: sqrt( ||V†V - I||_F^2 / ||I||_F^2 )
  auto unitarityResidual = [](const LatticeColourMatrixD& v) -> RealD {
    LatticeColourMatrixD vdv(v.Grid()), identity(v.Grid());
    identity = 1.0;
    vdv = adj(v)*v - identity;
    return std::sqrt(norm2(vdv) / norm2(identity));
  };

  const RealD stressTol = 1e-10;
  LatticeColourMatrixD su(&Grid), sv(&Grid);
  ColourMatrixD sm;

  // Run all six singular-link projection stress tests for a given projector.
  auto singularLinkStressTest = [&](
    const std::string&                 prefix,
    UnitaryProjection<PeriodicGimplD>& p
  ) {
    std::cout << GridLogMessage << "=== singular link stress test: " << prefix << " ===" << std::endl;

    // case 1: one near-zero singular value (sigma_min = 1e-10)
    sm = Zero();
    sm()()(0,0) = ComplexD(1.0); sm()()(1,1) = ComplexD(1.0); sm()()(2,2) = ComplexD(1e-10);
    fillField(su, sm);
    p.project(sv, su);
    {
      RealD res = unitarityResidual(sv);
      if (res < stressTol) Grid_pass("sigma={1,1,1e-10} (diag)          : ||VdagV-I||/||I|| = ", res);
      else                 Grid_error("sigma={1,1,1e-10} (diag)          : ||VdagV-I||/||I|| = ", res);
    }

    // case 2: rank-2 link (sigma_min = 0 exactly)
    sm = Zero();
    sm()()(0,0) = ComplexD(1.0); sm()()(1,1) = ComplexD(1.0);
    fillField(su, sm);
    p.project(sv, su);
    {
      RealD res = unitarityResidual(sv);
      if (res < stressTol) Grid_pass("sigma={1,1,0}    (rank-2)          : ||VdagV-I||/||I|| = ", res);
      else                 Grid_error("sigma={1,1,0}    (rank-2)          : ||VdagV-I||/||I|| = ", res);
    }

    // case 3: rank-1 link (two zero singular values)
    sm = Zero();
    sm()()(0,0) = ComplexD(1.0);
    fillField(su, sm);
    p.project(sv, su);
    {
      RealD res = unitarityResidual(sv);
      if (res < stressTol) Grid_pass("sigma={1,0,0}    (rank-1)          : ||VdagV-I||/||I|| = ", res);
      else                 Grid_error("sigma={1,0,0}    (rank-1)          : ||VdagV-I||/||I|| = ", res);
    }

    // case 4: zero matrix (all singular values = 0)
    sm = Zero();
    fillField(su, sm);
    p.project(sv, su);
    {
      RealD res = unitarityResidual(sv);
      if (res < stressTol) Grid_pass("sigma={0,0,0}    (zero link)       : ||VdagV-I||/||I|| = ", res);
      else                 Grid_error("sigma={0,0,0}    (zero link)       : ||VdagV-I||/||I|| = ", res);
    }

    // case 5: two degenerate near-zero singular values (sigma_1 = sigma_2 = 1e-10)
    sm = Zero();
    sm()()(0,0) = ComplexD(1.0); sm()()(1,1) = ComplexD(1e-10); sm()()(2,2) = ComplexD(1e-10);
    fillField(su, sm);
    p.project(sv, su);
    {
      RealD res = unitarityResidual(sv);
      if (res < stressTol) Grid_pass("sigma={1,1e-10,1e-10} (degenerate) : ||VdagV-I||/||I|| = ", res);
      else                 Grid_error("sigma={1,1e-10,1e-10} (degenerate) : ||VdagV-I||/||I|| = ", res);
    }

    // case 6: non-diagonal near-singular:  P * diag(1, 1, 1e-10) * P†
    //         where P is the cyclic permutation matrix
    {
      ColourMatrixD P = Zero(), D = Zero();
      P()()(0,1) = ComplexD(1.0); P()()(1,2) = ComplexD(1.0); P()()(2,0) = ComplexD(1.0);
      D()()(0,0) = ComplexD(1.0); D()()(1,1) = ComplexD(1.0); D()()(2,2) = ComplexD(1e-10);
      sm = P * D * adj(P);
      fillField(su, sm);
      p.project(sv, su);
      RealD res = unitarityResidual(sv);
      if (res < stressTol) Grid_pass("sigma={1,1,1e-10} (non-diag)      : ||VdagV-I||/||I|| = ", res);
      else                 Grid_error("sigma={1,1,1e-10} (non-diag)      : ||VdagV-I||/||I|| = ", res);
    }
  };

  singularLinkStressTest("Cayley-Hamilton + backup SVD", proj);

  {
    UnitaryProjectionContext ctxSVDProj(SingularValueDecompositionProjection, MIMDCollaborationDerivative);
    UnitaryProjection<PeriodicGimplD> projSVD(ctxSVDProj);
    singularLinkStressTest("pure SVD projection", projSVD);
  }

  /* derivative stress test */

  // Strategy: use the linear functional
  //   S(U) = -Re Tr(Z† V(U))  summed over mu and lattice sites,
  // where Z is a fixed random complex field and V(U) = proj(U).
  //
  // Convention: derivative() takes the upstream gradient G_V where
  //   delta S = Re Tr(G_V† delta V),
  // and returns the downstream gradient G_U where
  //   delta S = Re Tr(G_U† delta U).
  // For S = -Re Tr(Z†V), the gradient is G_V = -Z (not -Z†):
  //   Re Tr((-Z)† delta V) = Re Tr(-Z† delta V) = delta S  ✓
  // Hence:
  //   dzdv  = -Z_mu   (upstream gradient, no adjoint)
  //   pred  = Re Tr(adj(dvdu) * dU)  (adjoint of output contracted with dU)
  //
  // A correct derivative gives |S(U+eps*dU) - S(U) - eps*pred| = O(eps^2),
  // so err(eps)/err(eps/2) -> 4.
  //
  // SVD-fallback regime: hot config with absoluteSVDTolerance = 1.0, so SVD always
  // fires.  _bound cutoff = 1e-20 (essentially off).  Tests _eigs3SVD eigenvalue
  // extraction and the derivative formula in the SVD-fallback branch.

  std::cout << GridLogMessage << "=== derivative stress test ===" << std::endl;

  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers({1,2,3,4});

  LatticeGaugeField Zfield(&Grid);
  for (int mu = 0; mu < Nd; ++mu) {
    LatticeColourMatrixD Zmu(&Grid);
    gaussian(pRNG, Zmu);
    PokeIndex<LorentzIndex>(Zfield, Zmu, mu);
  }

  LatticeGaugeField dUpert(&Grid);
  for (int mu = 0; mu < Nd; ++mu) {
    LatticeColourMatrixD dUmu(&Grid);
    gaussian(pRNG, dUmu);
    PokeIndex<LorentzIndex>(dUpert, dUmu, mu);
  }

  // S(V) = -Re Tr(Z† V) summed over mu and sites
  auto computeS = [&](const LatticeGaugeField& V) -> RealD {
    RealD s = 0.0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrixD Vmu = PeekIndex<LorentzIndex>(V, mu);
      LatticeColourMatrixD Zmu = PeekIndex<LorentzIndex>(Zfield, mu);
      s -= real(sum(trace(adj(Zmu) * Vmu)));
    }
    return s;
  };

  // Upstream gradient: G_V = -Z_mu satisfies delta S = Re Tr(G_V† delta V).
  auto computedSdV = [&]() -> LatticeGaugeField {
    LatticeGaugeField dSdV(&Grid);
    for (int mu = 0; mu < Nd; ++mu)
      PokeIndex<LorentzIndex>(dSdV, -PeekIndex<LorentzIndex>(Zfield, mu), mu);
    return dSdV;
  };

  // Perturb U0 by eps in the direction dU
  auto perturb = [&](const LatticeGaugeField& U0, RealD eps) -> LatticeGaugeField {
    LatticeGaugeField Up(U0.Grid());
    autoView(U0_v, U0,     CpuRead);
    autoView(dU_v, dUpert, CpuRead);
    autoView(Up_v, Up,     CpuWrite);
    thread_foreach(i, Up_v, {
      for (int mu = 0; mu < Nd; ++mu)
        Up_v[i](mu) = U0_v[i](mu) + eps * dU_v[i](mu);
    });
    return Up;
  };

  // Finite-difference check: report convergence ratio and pass/fail
  auto fdCheck = [&](
    const std::string&                  label,
    const LatticeGaugeField&            U0,
    UnitaryProjection<PeriodicGimplD>&  proj,
    RealD                               dt
  ) {
    LatticeGaugeField V0(U0.Grid());
    proj.project(V0, U0);
    RealD S0 = computeS(V0);

    // upstream gradient dS/dV, chain-ruled to dS/dU through projection
    LatticeGaugeField dSdV = computedSdV();
    LatticeGaugeField dSdU(U0.Grid()), dSdU_jo(U0.Grid());
    proj.derivative(dSdU, dSdV, U0);
    // also test JO derivative for comparison
    {
      UnitaryProjectionContext ctxJO(CayleyHamiltonProjection, JinOsbornDerivative);
      UnitaryProjection<PeriodicGimplD> projJO(ctxJO);
      projJO.derivative(dSdU_jo, dSdV, V0, U0);
    }

    // linear prediction: pred = Re Tr(adj(dvdu) * dU)  [convention: dvdu satisfies delta S = Re Tr(dvdu† * delta U)]
    RealD pred  = 0.0, predJO = 0.0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrixD gmu    = PeekIndex<LorentzIndex>(dSdU,    mu);
      LatticeColourMatrixD gmu_jo = PeekIndex<LorentzIndex>(dSdU_jo, mu);
      LatticeColourMatrixD dumu   = PeekIndex<LorentzIndex>(dUpert,  mu);
      pred   += real(sum(trace(adj(gmu)    * dumu)));
      predJO += real(sum(trace(adj(gmu_jo) * dumu)));
    }
    // also try pred from direct formula using Z and dU
    RealD predExact = 0.0;
    for (int mu = 0; mu < Nd; ++mu) {
      LatticeColourMatrixD Vmu  = PeekIndex<LorentzIndex>(V0,     mu);
      LatticeColourMatrixD Zmu  = PeekIndex<LorentzIndex>(Zfield, mu);
      LatticeColourMatrixD dumu = PeekIndex<LorentzIndex>(dUpert, mu);
      predExact += 0.5*real(sum(trace((adj(Vmu)*Zmu*adj(Vmu) - adj(Zmu))*dumu)));
    }

    std::vector<RealD> errs;
    std::vector<RealD> Svals;
    for (RealD eps : {dt, dt/2.0, dt/4.0}) {
      LatticeGaugeField Veps(U0.Grid());
      proj.project(Veps, perturb(U0, eps));
      Svals.push_back(computeS(Veps));
      errs.push_back(std::abs(computeS(Veps) - S0 - eps*pred));
    }

    // exact numerical slope (1st-order FD)
    RealD exactSlope = (Svals[0] - S0) / dt;
    std::cout << GridLogMessage << "  pred()=" << pred << "  pred(JO)=" << predJO
              << "  predExact=" << predExact << "  exactSlope=" << exactSlope << std::endl;

    RealD ratio10 = (errs[1] > 0) ? errs[0]/errs[1] : 0.0;
    RealD ratio21 = (errs[2] > 0) ? errs[1]/errs[2] : 0.0;

    std::cout << GridLogMessage << label
              << "  err(dt)="    << errs[0]
              << "  err(dt/2)="  << errs[1]
              << "  ratio10="    << ratio10
              << "  ratio21="    << ratio21
              << " (expect ~4)"  << std::endl;

    bool ok = ratio10 > 2.0 && ratio10 < 8.0 && ratio21 > 2.0 && ratio21 < 8.0;
    if (ok) Grid_pass(label, ": O(dt^2) convergence");
    else    Grid_error(label, ": O(dt^2) convergence");
  };

  // Links: cold config with mu=0 links replaced by diag(1, 1, 1e-4).
  // absoluteSVDTolerance = 1.0: fires because e_min = (1e-4)^2 = 1e-8 < 1.0.
  // _bound cutoff = 1e-20: essentially off, so eigenvalues are exact SVD values.
  {
    UnitaryProjectionContext ctxSVDTest(CayleyHamiltonProjection, MIMDCollaborationDerivative);
    ctxSVDTest.setDerivativeEigenvalueCutoff(1e-20);  // _bound essentially off
    ctxSVDTest.setBackupSVD(true);
    ctxSVDTest.setRelativeSVDTolerance(0.0);          // never trigger via det check
    ctxSVDTest.setAbsoluteSVDTolerance(1.0);          // always trigger via eigenvalue check
    UnitaryProjection<PeriodicGimplD> projSVDTest(ctxSVDTest);

    LatticeGaugeField U0svd(&Grid);
    SU<Nc>::HotConfiguration(pRNG, U0svd);

    // these should give similar results
    fdCheck("SVD-fallback  (sigma_min=1e-4, absTol=1.0)", U0svd, proj, 1e-3);
    fdCheck("SVD-fallback  (sigma_min=1e-4, absTol=1.0)", U0svd, projSVDTest, 1e-3);
  }

  // svdOnlyDerivative path: _eigs3SVD used exclusively for eigenvalue extraction inside
  // _derivativeU3MILC, bypassing _eigs3 + backup entirely.  CH projection for forward pass.
  {
    UnitaryProjectionContext ctxSVDDeriv(CayleyHamiltonProjection, MIMDCollaborationDerivative);
    ctxSVDDeriv.setSVDOnlyDerivative(true);
    UnitaryProjection<PeriodicGimplD> projSVDDeriv(ctxSVDDeriv);

    LatticeGaugeField U0(&Grid);
    SU<Nc>::HotConfiguration(pRNG, U0);

    fdCheck("svdOnlyDerivative (CH projection)", U0, projSVDDeriv, 1e-3);
  }

  // Pure SVD path: _projectU3SVD for projection, _eigs3SVD for derivative eigenvalues.
  {
    UnitaryProjectionContext ctxPureSVD(SingularValueDecompositionProjection, MIMDCollaborationDerivative);
    ctxPureSVD.setSVDOnlyDerivative(true);
    UnitaryProjection<PeriodicGimplD> projPureSVD(ctxPureSVD);

    LatticeGaugeField U0(&Grid);
    SU<Nc>::HotConfiguration(pRNG, U0);

    fdCheck("SVD projection + svdOnlyDerivative", U0, projPureSVD, 1e-3);
  }

  /* finalize */

  std::cout<< GridLogMessage << "Done" <<std::endl;
  Grid_finalize();
};