/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/utils/HISQStencilKernels.h
Author: D. A. Clarke <clarke.davida@gmail.com>
Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

Copyright (C) 2024

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
 * @file HISQStencilKernels.h
 * @brief Stencil-based GPU kernels for HISQ smearing and its derivative
 * @author David Clarke (primary) and Curtis Taylor Peterson (port)
 * @details
 * GPU-optimized kernels for HISQ fat7/asqtad smearing and derivative using
 * GeneralLocalStencil. These kernels avoid Grid's expression template machinery
 * by pre-computing all stencil shifts and performing link operations inside
 * single accelerator_for loops. This eliminates the millions of tiny cudaMemcpy
 * calls that expression templates generate on GPU.
 *
 * The stencil approach works by:
 * 1. Exchanging the gauge field into a PaddedCell (single halo exchange)
 * 2. Pre-computing all multi-hop shift coordinates via GeneralLocalStencil
 * 3. Reading shifted links inside accelerator_for kernels using stencil entries
 * 4. Accumulating results with coalescedWrite
 *
 * Halo depth of 1 is sufficient because each dimension is shifted by at most
 * +/-1 (even for 7-link paths, each direction independently shifts by +/-1).
 *
 * References:
 * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
 */

#pragma once

#ifndef QCD_UTILS_HISQ_STENCIL_KERNELS_H
#define QCD_UTILS_HISQ_STENCIL_KERNELS_H

#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>

NAMESPACE_BEGIN(Grid);

// 
// Stencil utility functions
// 

// Append a compound shift to the shift vector
template<typename... Args>
inline void hisqAppendShift(std::vector<Coordinate>& shifts, int dir, Args... args) {
  Coordinate shift(Nd, 0);
  generalShift(shift, dir, args...);
  shifts.push_back(shift);
}

// Stencil index for 3-link stencil (5 entries per mu-nu pair)
accelerator_inline int hisqStencilIndex3(int mu, int nu) { return 5*(nu + Nd*mu); }

// Stencil index for 5-link stencil (17 entries per mu-nu-rho triplet)
accelerator_inline int hisqStencilIndex5(int mu, int nu, int rho) 
{ return 17*(rho + Nd*nu + Nd*Nd*mu); }

// Stencil index for 7-link stencil (46 entries per mu-nu-rho-sig quad)
accelerator_inline int hisqStencilIndex7(int mu, int nu, int rho, int sig) 
{ return 46*(sig + Nd*rho + Nd*Nd*nu + Nd*Nd*Nd*mu); }

// Read a link from a stencil-accessed view
template<class LinkView> accelerator_inline
auto link(const LinkView& __restrict__ U, GeneralStencilEntry* x, int mu) 
{ return coalescedReadGeneralPermute(U[x->_offset](mu), x->_permute, Nd); }

// 
// Stencil construction
// 

// Create shift vectors for HISQ stencils
// @param kind: 3 for 3-link, 5 for 5-link, 7 for 7-link
inline std::vector<Coordinate> createHISQStencil(int kind) {
    std::vector<Coordinate> shifts;

    if (kind == 3) {
        // 5 entries per (mu, nu) pair: mu, nu, 0, mu-nu, -nu
        for (int mu = 0; mu < Nd; mu++)
        for (int nu = 0; nu < Nd; nu++) {
            hisqAppendShift(shifts, mu);
            hisqAppendShift(shifts, nu);
            hisqAppendShift(shifts, shiftSignal::NO_SHIFT);
            hisqAppendShift(shifts, mu, Back(nu));
            hisqAppendShift(shifts, Back(nu));
        }
    } else if (kind == 5) {
        // 17 entries per (mu, nu, rho) triplet
        for (int mu = 0; mu < Nd; mu++)
        for (int nu = 0; nu < Nd; nu++)
        for (int rho = 0; rho < Nd; rho++) {
            hisqAppendShift(shifts, nu, Back(rho));           // 0
            hisqAppendShift(shifts, nu);                      // 1
            hisqAppendShift(shifts, Back(rho));               // 2
            hisqAppendShift(shifts, shiftSignal::NO_SHIFT);   // 3
            hisqAppendShift(shifts, rho);                     // 4
            hisqAppendShift(shifts, Back(nu), Back(rho));     // 5
            hisqAppendShift(shifts, Back(nu));                // 6
            hisqAppendShift(shifts, Back(nu), rho);           // 7
            hisqAppendShift(shifts, mu, nu, Back(rho));       // 8
            hisqAppendShift(shifts, mu, nu);                  // 9
            hisqAppendShift(shifts, mu, Back(rho));           // 10
            hisqAppendShift(shifts, mu);                      // 11
            hisqAppendShift(shifts, mu, rho);                 // 12
            hisqAppendShift(shifts, mu, Back(nu), Back(rho)); // 13
            hisqAppendShift(shifts, mu, Back(nu));            // 14
            hisqAppendShift(shifts, mu, Back(nu), rho);       // 15
            hisqAppendShift(shifts, nu, rho);                 // 16
        }
    } else if (kind == 7) {
        // 46 entries per (mu, nu, rho, sig) quad
        for (int mu = 0; mu < Nd; mu++)
        for (int nu = 0; nu < Nd; nu++)
        for (int rho = 0; rho < Nd; rho++)
        for (int sig = 0; sig < Nd; sig++) {
            hisqAppendShift(shifts, shiftSignal::NO_SHIFT);              // 0
            hisqAppendShift(shifts, mu);                                 // 1
            hisqAppendShift(shifts, mu, nu);                             // 2
            hisqAppendShift(shifts, mu, nu, rho);                        // 3
            hisqAppendShift(shifts, mu, nu, rho, Back(sig));             // 4
            hisqAppendShift(shifts, mu, nu, Back(rho));                  // 5
            hisqAppendShift(shifts, mu, nu, Back(rho), Back(sig));       // 6
            hisqAppendShift(shifts, mu, nu, Back(sig));                  // 7
            hisqAppendShift(shifts, mu, Back(nu));                       // 8
            hisqAppendShift(shifts, mu, Back(nu), rho);                  // 9
            hisqAppendShift(shifts, mu, Back(nu), rho, sig);             // 10
            hisqAppendShift(shifts, mu, Back(nu), rho, Back(sig));       // 11
            hisqAppendShift(shifts, mu, Back(nu), Back(rho));            // 12
            hisqAppendShift(shifts, mu, Back(nu), Back(rho), sig);       // 13
            hisqAppendShift(shifts, mu, Back(nu), Back(rho), Back(sig)); // 14
            hisqAppendShift(shifts, mu, Back(nu), sig);                  // 15
            hisqAppendShift(shifts, mu, Back(nu), Back(sig));            // 16
            hisqAppendShift(shifts, mu, rho);                            // 17
            hisqAppendShift(shifts, mu, rho, sig);                       // 18
            hisqAppendShift(shifts, mu, rho, Back(sig));                 // 19
            hisqAppendShift(shifts, mu, Back(rho));                      // 20
            hisqAppendShift(shifts, mu, Back(rho), sig);                 // 21
            hisqAppendShift(shifts, mu, Back(rho), Back(sig));           // 22
            hisqAppendShift(shifts, mu, sig);                            // 23
            hisqAppendShift(shifts, mu, Back(sig));                      // 24
            hisqAppendShift(shifts, nu);                                 // 25
            hisqAppendShift(shifts, nu, rho);                            // 26
            hisqAppendShift(shifts, nu, rho, sig);                       // 27
            hisqAppendShift(shifts, nu, rho, Back(sig));                 // 28
            hisqAppendShift(shifts, nu, Back(rho));                      // 29
            hisqAppendShift(shifts, nu, Back(rho), sig);                 // 30
            hisqAppendShift(shifts, nu, Back(rho), Back(sig));           // 31
            hisqAppendShift(shifts, rho);                                // 32
            hisqAppendShift(shifts, rho, Back(nu));                      // 33
            hisqAppendShift(shifts, rho, sig);                           // 34
            hisqAppendShift(shifts, rho, Back(sig));                     // 35
            hisqAppendShift(shifts, Back(nu));                           // 36
            hisqAppendShift(shifts, Back(nu), rho);                      // 37
            hisqAppendShift(shifts, Back(nu), rho, sig);                 // 38
            hisqAppendShift(shifts, Back(nu), rho, Back(sig));           // 39
            hisqAppendShift(shifts, Back(nu), Back(rho));                // 40
            hisqAppendShift(shifts, Back(nu), Back(rho), sig);           // 41
            hisqAppendShift(shifts, Back(nu), Back(rho), Back(sig));     // 42
            hisqAppendShift(shifts, Back(rho));                          // 43
            hisqAppendShift(shifts, Back(rho), sig);                     // 44
            hisqAppendShift(shifts, Back(rho), Back(sig));               // 45
        }
    }
    return shifts;
}

// 
// Smearing kernels
// 

// Three-link staple kernel: accumulates 3-link contribution to fat links
// and stores the 3-link construct for higher-order staples.
template<class GF>
void hisqThreeLinkStaple(
  GF& U_fat, 
  GF& U_3link, 
  GF& U,
  GeneralLocalStencil& gStencil, 
  int mu, 
  RealD c3
) {
  autoView(U_v      , U      , AcceleratorRead);
  autoView(U_fat_v  , U_fat  , AcceleratorWrite);
  autoView(U_3link_v, U_3link, AcceleratorWrite);
  auto gStencil_v = gStencil.View(AcceleratorRead);
  int Nsites = U_v.size();
  int Nsimd  = U.Grid()->Nsimd();

  typedef decltype(link(U_v, gStencil_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Nsimd, {
    U3matrix res;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;
      int s = hisqStencilIndex3(mu, nu);

      auto x_p_mu      = gStencil_v.GetEntry(s+0, site);
      auto x_p_nu      = gStencil_v.GetEntry(s+1, site);
      auto x           = gStencil_v.GetEntry(s+2, site);
      auto x_p_mu_m_nu = gStencil_v.GetEntry(s+3, site);
      auto x_m_nu      = gStencil_v.GetEntry(s+4, site);

      res = link(U_v, x, nu)*link(U_v, x_p_nu, mu)*adj(link(U_v, x_p_mu, nu))
          + adj(link(U_v, x_m_nu, nu))*link(U_v, x_m_nu, mu)*link(U_v, x_p_mu_m_nu, nu);

      coalescedWrite(U_3link_v[x->_offset](nu), res);
      coalescedWrite(U_fat_v[x->_offset](mu), U_fat_v(x->_offset)(mu) + c3*res);
  } } )
}

// Five-link staple kernel: accumulates 5-link contribution to fat links
// and stores the 5-link constructs for the 7-link staple.
template<class GF>
void hisqFiveLinkStaple(
  GF& U_fat, 
  GF& U_5linkA, 
  GF& U_5linkB, 
  GF& U_3link,
  GF& U, 
  GeneralLocalStencil& gStencil, 
  int mu, 
  RealD c5
) {
  autoView(U_v       , U       , AcceleratorRead);
  autoView(U_fat_v   , U_fat   , AcceleratorWrite);
  autoView(U_3link_v , U_3link , AcceleratorRead);
  autoView(U_5linkA_v, U_5linkA, AcceleratorWrite);
  autoView(U_5linkB_v, U_5linkB, AcceleratorWrite);
  auto gStencil_v = gStencil.View(AcceleratorRead);
  int Nsites = U_v.size();
  int Nsimd  = U.Grid()->Nsimd();

  typedef decltype(link(U_v, gStencil_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Nsimd, {
    U3matrix res;
    int sigmaIndex = 0;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;
      int s = hisqStencilIndex3(mu, nu);
      for (int rho = 0; rho < Nd; rho++) {
        if (rho == mu || rho == nu) continue;

        auto x_p_mu      = gStencil_v.GetEntry(s+0, site);
        auto x_p_nu      = gStencil_v.GetEntry(s+1, site);
        auto x           = gStencil_v.GetEntry(s+2, site);
        auto x_p_mu_m_nu = gStencil_v.GetEntry(s+3, site);
        auto x_m_nu      = gStencil_v.GetEntry(s+4, site);

        res = link(U_v, x, nu)  * link(U_3link_v, x_p_nu, rho) * adj(link(U_v, x_p_mu, nu)) 
            + adj(link(U_v, x_m_nu, nu))*link(U_3link_v, x_m_nu, rho)*link(U_v, x_p_mu_m_nu, nu);

        if (sigmaIndex < 3) { coalescedWrite(U_5linkA_v[x->_offset](rho), res); }
        else { coalescedWrite(U_5linkB_v[x->_offset](rho), res); }

        coalescedWrite(U_fat_v[x->_offset](mu), U_fat_v(x->_offset)(mu) + c5*res);
        sigmaIndex++;
  } } } )
}

// Seven-link staple kernel: accumulates 7-link contribution to fat links.
template<class GF>
void hisqSevenLinkStaple(
  GF& U_fat, 
  GF& U_5linkA, 
  GF& U_5linkB,
  GF& U, 
  GeneralLocalStencil& gStencil, 
  int mu, 
  RealD c7
) {
  autoView(U_v       , U       , AcceleratorRead);
  autoView(U_fat_v   , U_fat   , AcceleratorWrite);
  autoView(U_5linkA_v, U_5linkA, AcceleratorRead);
  autoView(U_5linkB_v, U_5linkB, AcceleratorRead);
  auto gStencil_v = gStencil.View(AcceleratorRead);
  int Nsites = U_v.size();
  int Nsimd  = U.Grid()->Nsimd();

  typedef decltype(link(U_v, gStencil_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Nsimd, {
    U3matrix res;
    int sigmaIndex = 0;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;
      
      int s = hisqStencilIndex3(mu, nu);
      for (int rho = 0; rho < Nd; rho++) {
        if (rho == mu || rho == nu) continue;

        auto x_p_mu      = gStencil_v.GetEntry(s+0, site);
        auto x_p_nu      = gStencil_v.GetEntry(s+1, site);
        auto x           = gStencil_v.GetEntry(s+2, site);
        auto x_p_mu_m_nu = gStencil_v.GetEntry(s+3, site);
        auto x_m_nu      = gStencil_v.GetEntry(s+4, site);

        if (sigmaIndex < 3) {
          res = link(U_v, x, nu)*link(U_5linkB_v, x_p_nu, rho)*adj(link(U_v, x_p_mu, nu))
              + adj(link(U_v, x_m_nu, nu))*link(U_5linkB_v, x_m_nu, rho)*link(U_v, x_p_mu_m_nu, nu);
        }
        else {
          res = link(U_v, x, nu)*link(U_5linkA_v, x_p_nu, rho)*adj(link(U_v, x_p_mu, nu))
              + adj(link(U_v, x_m_nu, nu))*link(U_5linkA_v, x_m_nu, rho)*link(U_v, x_p_mu_m_nu, nu);
        }
        
        coalescedWrite(U_fat_v[x->_offset](mu), U_fat_v(x->_offset)(mu) + c7*res);
        sigmaIndex++;
  } } } )
}

// 
// Derivative kernels
// 

// Three-link derivative kernel: accumulates 3-link chain rule contribution
// to the force. For each mu, sums over nu the 6 terms from differentiating
// the upper and lower 3-link staples with respect to each of the 3 links.
template<class GF>
void hisqThreeLinkDeriv(
  GF& Fghost, 
  GF& Ughost, 
  GF& XYghost,
  GeneralLocalStencil& gStencil3, 
  RealD c3, 
  int mu
) {
  autoView(U_v , Ughost , AcceleratorRead);
  autoView(XY_v, XYghost, AcceleratorRead);
  autoView(F_v , Fghost , AcceleratorWrite);
  int Nsites = U_v.size();
  auto gStencil3_v = gStencil3.View(AcceleratorRead);

  typedef decltype(link(U_v, gStencil3_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Fghost.Grid()->Nsimd(), {
    U3matrix res;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;
      int s = hisqStencilIndex3(mu, nu);

      auto x_p_mu      = gStencil3_v.GetEntry(s+0, site);
      auto x_p_nu      = gStencil3_v.GetEntry(s+1, site);
      auto x           = gStencil3_v.GetEntry(s+2, site);
      auto x_p_mu_m_nu = gStencil3_v.GetEntry(s+3, site);
      auto x_m_nu      = gStencil3_v.GetEntry(s+4, site);

      res = adj(link(XY_v, x, nu))*link(U_v,  x_p_nu, mu)*adj(link(U_v, x_p_mu, nu))
          + link(U_v, x, nu)*adj(link(XY_v, x_p_nu, mu))*adj(link(U_v, x_p_mu, nu))
          + link(U_v, x, nu)*link(U_v, x_p_nu, mu)*link(XY_v, x_p_mu, nu)
          + link(XY_v, x_m_nu, nu)*link(U_v,  x_m_nu, mu)*link(U_v,  x_p_mu_m_nu, nu)
          + adj(link(U_v, x_m_nu, nu))*adj(link(XY_v, x_m_nu, mu))*link(U_v,  x_p_mu_m_nu, nu)
          + adj(link(U_v, x_m_nu, nu))*link(U_v, x_m_nu, mu)*adj(link(XY_v, x_p_mu_m_nu, nu));

      coalescedWrite(F_v[x->_offset](mu), F_v(x->_offset)(mu) + c3*adj(res));
  } } )
}

// Five-link derivative kernel with compile-time term selection.
// term=0 and term=1 each handle half of the 5-link contributions.
template<int term, class GF>
void hisqFiveLinkDeriv(
  GF& Fghost, 
  GF& Ughost, 
  GF& XYghost,
  GeneralLocalStencil& gStencil5, 
  RealD c5, 
  int mu
) {
  autoView(U_v , Ughost , AcceleratorRead);
  autoView(XY_v, XYghost, AcceleratorRead);
  autoView(F_v , Fghost , AcceleratorWrite);
  int Nsites = U_v.size();
  auto gStencil5_v = gStencil5.View(AcceleratorRead);

  typedef decltype(link(U_v, gStencil5_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Fghost.Grid()->Nsimd(), {
    U3matrix res;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;
      for (int rho = 0; rho < Nd; rho++) {
        if (rho == mu || rho == nu) continue;
        
        int s = hisqStencilIndex5(mu, nu, rho);

        auto x_p_nu_m_rho      = gStencil5_v.GetEntry(s+0 , site);
        auto x_p_nu            = gStencil5_v.GetEntry(s+1 , site);
        auto x_m_rho           = gStencil5_v.GetEntry(s+2 , site);
        auto x                 = gStencil5_v.GetEntry(s+3 , site);
        auto x_p_rho           = gStencil5_v.GetEntry(s+4 , site);
        auto x_m_nu_m_rho      = gStencil5_v.GetEntry(s+5 , site);
        auto x_m_nu            = gStencil5_v.GetEntry(s+6 , site);
        auto x_m_nu_p_rho      = gStencil5_v.GetEntry(s+7 , site);
        auto x_p_mu_p_nu_m_rho = gStencil5_v.GetEntry(s+8 , site);
        auto x_p_mu_p_nu       = gStencil5_v.GetEntry(s+9 , site);
        auto x_p_mu_m_rho      = gStencil5_v.GetEntry(s+10, site);
        auto x_p_mu            = gStencil5_v.GetEntry(s+11, site);
        auto x_p_mu_p_rho      = gStencil5_v.GetEntry(s+12, site);
        auto x_p_mu_m_nu_m_rho = gStencil5_v.GetEntry(s+13, site);
        auto x_p_mu_m_nu       = gStencil5_v.GetEntry(s+14, site);
        auto x_p_mu_m_nu_p_rho = gStencil5_v.GetEntry(s+15, site);
        auto x_p_nu_p_rho      = gStencil5_v.GetEntry(s+16, site);

        res = Zero();

        if constexpr(term==0) {
        res += (link(U_v, x_p_mu, rho)
               *link(U_v, x_p_mu_p_rho, nu)
               *adj(link(U_v, x_p_mu_p_nu, rho))
               +adj(link(U_v, x_p_mu_m_rho, rho))
               *link(U_v, x_p_mu_m_rho, nu)
               *link(U_v, x_p_mu_p_nu_m_rho, rho)
               )*adj(link(U_v, x_p_nu, mu))*link(XY_v, x, nu);

        res += (link(U_v, x_p_mu, rho)
               *adj(link(U_v, x_p_rho, mu))
               *adj(link(U_v, x_m_nu_p_rho, nu))
               *link(XY_v, x_m_nu, rho)
               +adj(link(U_v, x_p_mu_m_rho, rho))
               * adj(link(U_v, x_m_rho, mu))
               * adj(link(U_v, x_m_nu_m_rho, nu))
               * adj(link(XY_v, x_m_nu_m_rho, rho))
               )*link(U_v, x_m_nu, nu);

        res += (link(U_v, x_p_mu, rho)
               *adj(link(U_v, x_p_mu_m_nu_p_rho, nu))
               *adj(link(U_v, x_p_mu_m_nu, rho))
               +adj(link(U_v, x_p_mu_m_rho, rho))
               *adj(link(U_v, x_p_mu_m_nu_m_rho, nu))
               *link(U_v, x_p_mu_m_nu_m_rho, rho)
               )*adj(link(U_v, x_m_nu, mu))*adj(link(XY_v, x_m_nu, nu));

        res += (link(U_v, x_p_mu, rho)
               *adj(link(U_v, x_p_rho, mu))
               *link(U_v, x_p_rho, nu)
               *link(XY_v, x_p_nu, rho)
               +adj(link(U_v, x_p_mu_m_rho, rho))
               *adj(link(U_v, x_m_rho, mu))
               *link(U_v, x_m_rho, nu)
               *adj(link(XY_v, x_p_nu_m_rho, rho))
               )*adj(link(U_v, x, nu));

        res += (link(XY_v, x_p_mu_m_nu, nu)
               *adj(link(U_v, x_m_nu, mu))
               *link(U_v, x_m_nu, rho)
               +link(U_v, x_p_mu, rho)
               *adj(link(U_v, x_p_mu_m_nu_p_rho, nu))
               *link(XY_v, x_m_nu_p_rho, mu)
               )*link(U_v, x_m_nu_p_rho, nu)*adj(link(U_v, x, rho));
        }
        if constexpr(term==1) {
          res += (link(U_v, x_p_mu, nu)
                  *adj(link(XY_v, x_p_mu_p_nu, rho))
                  *adj(link(U_v, x_p_mu_p_rho, nu))
                  +adj(link(U_v, x_p_mu_m_nu, nu))
                  *adj(link(XY_v, x_p_mu_m_nu, rho))
                  *link(U_v, x_p_mu_m_nu_p_rho, nu)
                  )*adj(link(U_v, x_p_rho, mu))*adj(link(U_v, x, rho));

          res += (link(U_v, x_p_mu, rho)
                  *link(U_v, x_p_mu_p_rho, nu)
                  *link(XY_v, x_p_nu_p_rho, mu)
                  +adj(link(XY_v, x_p_mu, nu))
                  *adj(link(U_v, x_p_nu, mu))
                  *link(U_v, x_p_nu, rho)
                )*adj(link(U_v, x_p_rho, nu))*adj(link(U_v, x, rho));

          res += (adj(link(XY_v, x_p_mu, nu))
                  *adj(link(U_v, x_p_nu, mu))
                  *adj(link(U_v, x_p_nu_m_rho, rho))
                  +adj(link(U_v, x_p_mu_m_rho, rho))
                  *link(U_v, x_p_mu_m_rho, nu)
                  *link(XY_v, x_p_nu_m_rho, mu)
                  )*adj(link(U_v, x_m_rho, nu))*link(U_v, x_m_rho, rho);

          res += (link(U_v, x_p_mu, nu)
                  *link(XY_v, x_p_mu_p_nu_m_rho, rho)
                  *adj(link(U_v, x_p_mu_m_rho, nu))
                  +adj(link(U_v, x_p_mu_m_nu, nu))
                  *link(XY_v, x_p_mu_m_nu_m_rho, rho)
                  *link(U_v, x_p_mu_m_nu_m_rho, nu)
                  )*adj(link(U_v, x_m_rho, mu))*link(U_v, x_m_rho, rho);

          res += (link(XY_v, x_p_mu_m_nu, nu)
                  *adj(link(U_v, x_m_nu, mu))
                  *adj(link(U_v, x_m_nu_m_rho, rho))
                  +adj(link(U_v, x_p_mu_m_rho, rho))
                  *adj(link(U_v, x_p_mu_m_nu_m_rho, nu))
                  *link(XY_v, x_m_nu_m_rho, mu)
                  )*link(U_v, x_m_nu_m_rho, nu)*link(U_v, x_m_rho, rho);
        }

        coalescedWrite(F_v[x->_offset](mu), F_v(x->_offset)(mu) + c5*res);
  } } } )
}

// Seven-link derivative kernel with compile-time term selection.
// Terms 0-13 each handle a distinct contribution to the 7-link force.
template<int term, class GF>
void hisqSevenLinkDeriv(
  GF& Fghost,
  GF& Ughost,
  GF& XYghost,
  GeneralLocalStencil& gStencil7,
  RealD c7,
  int mu
) {
  autoView(U_v , Ughost , AcceleratorRead);
  autoView(XY_v, XYghost, AcceleratorRead);
  autoView(F_v , Fghost , AcceleratorWrite);
  int Nsites = U_v.size();
  auto gStencil7_v = gStencil7.View(AcceleratorRead);
  
  typedef decltype(link(U_v, gStencil7_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Fghost.Grid()->Nsimd(), {
    U3matrix res, U1;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;
      for (int rho = 0; rho < Nd; rho++) {
        if (rho == mu || rho == nu) continue;
        for (int sig = 0; sig < Nd; sig++) {
          if (sig == mu || sig == nu || sig == rho) continue;
          
          int s = hisqStencilIndex7(mu, nu, rho, sig);

          auto x                        = gStencil7_v.GetEntry(s+0 , site);
          auto x_p_mu                   = gStencil7_v.GetEntry(s+1 , site);
          auto x_p_mu_p_nu              = gStencil7_v.GetEntry(s+2 , site);
          auto x_p_mu_p_nu_p_rho        = gStencil7_v.GetEntry(s+3 , site);
          auto x_p_mu_p_nu_p_rho_m_sig  = gStencil7_v.GetEntry(s+4 , site);
          auto x_p_mu_p_nu_m_rho        = gStencil7_v.GetEntry(s+5 , site);
          auto x_p_mu_p_nu_m_rho_m_sig  = gStencil7_v.GetEntry(s+6 , site);
          auto x_p_mu_p_nu_m_sig        = gStencil7_v.GetEntry(s+7 , site);
          auto x_p_mu_m_nu              = gStencil7_v.GetEntry(s+8 , site);
          auto x_p_mu_m_nu_p_rho        = gStencil7_v.GetEntry(s+9 , site);
          auto x_p_mu_m_nu_p_rho_p_sig  = gStencil7_v.GetEntry(s+10, site);
          auto x_p_mu_m_nu_p_rho_m_sig  = gStencil7_v.GetEntry(s+11, site);
          auto x_p_mu_m_nu_m_rho        = gStencil7_v.GetEntry(s+12, site);
          auto x_p_mu_m_nu_m_rho_p_sig  = gStencil7_v.GetEntry(s+13, site);
          auto x_p_mu_m_nu_m_rho_m_sig  = gStencil7_v.GetEntry(s+14, site);
          auto x_p_mu_m_nu_p_sig        = gStencil7_v.GetEntry(s+15, site);
          auto x_p_mu_m_nu_m_sig        = gStencil7_v.GetEntry(s+16, site);
          auto x_p_mu_p_rho             = gStencil7_v.GetEntry(s+17, site);
          auto x_p_mu_p_rho_p_sig       = gStencil7_v.GetEntry(s+18, site);
          auto x_p_mu_p_rho_m_sig       = gStencil7_v.GetEntry(s+19, site);
          auto x_p_mu_m_rho             = gStencil7_v.GetEntry(s+20, site);
          auto x_p_mu_m_rho_p_sig       = gStencil7_v.GetEntry(s+21, site);
          auto x_p_mu_m_rho_m_sig       = gStencil7_v.GetEntry(s+22, site);
          auto x_p_mu_p_sig             = gStencil7_v.GetEntry(s+23, site);
          auto x_p_mu_m_sig             = gStencil7_v.GetEntry(s+24, site);
          auto x_p_nu                   = gStencil7_v.GetEntry(s+25, site);
          auto x_p_nu_p_rho             = gStencil7_v.GetEntry(s+26, site);
          auto x_p_nu_p_rho_p_sig       = gStencil7_v.GetEntry(s+27, site);
          auto x_p_nu_p_rho_m_sig       = gStencil7_v.GetEntry(s+28, site);
          auto x_p_nu_m_rho             = gStencil7_v.GetEntry(s+29, site);
          auto x_p_nu_m_rho_p_sig       = gStencil7_v.GetEntry(s+30, site);
          auto x_p_nu_m_rho_m_sig       = gStencil7_v.GetEntry(s+31, site);
          auto x_p_rho                  = gStencil7_v.GetEntry(s+32, site);
          auto x_p_rho_m_nu             = gStencil7_v.GetEntry(s+33, site);
          auto x_p_rho_p_sig            = gStencil7_v.GetEntry(s+34, site);
          auto x_p_rho_m_sig            = gStencil7_v.GetEntry(s+35, site);
          auto x_m_nu                   = gStencil7_v.GetEntry(s+36, site);
          auto x_m_nu_p_rho             = gStencil7_v.GetEntry(s+37, site);
          auto x_m_nu_p_rho_p_sig       = gStencil7_v.GetEntry(s+38, site);
          auto x_m_nu_p_rho_m_sig       = gStencil7_v.GetEntry(s+39, site);
          auto x_m_nu_m_rho             = gStencil7_v.GetEntry(s+40, site);
          auto x_m_nu_m_rho_p_sig       = gStencil7_v.GetEntry(s+41, site);
          auto x_m_nu_m_rho_m_sig       = gStencil7_v.GetEntry(s+42, site);
          auto x_m_rho                  = gStencil7_v.GetEntry(s+43, site);
          auto x_m_rho_p_sig            = gStencil7_v.GetEntry(s+44, site);
          auto x_m_rho_m_sig            = gStencil7_v.GetEntry(s+45, site);

          if constexpr(term==0) {
            res = adj(link(XY_v, x_p_mu, nu))*adj(link(U_v, x_p_nu, mu))
                  *(link(U_v, x_p_nu, rho)
                  *(link(U_v, x_p_nu_p_rho, sig)
                  *adj(link(U_v, x_p_rho_p_sig, nu))
                  *adj(link(U_v, x_p_rho, sig))
                  +adj(link(U_v, x_p_nu_p_rho_m_sig, sig))
                  *adj(link(U_v, x_p_rho_m_sig, nu))
                  *link(U_v, x_p_rho_m_sig, sig)
                  )*adj(link(U_v, x, rho))
                  +adj(link(U_v, x_p_nu_m_rho, rho))
                  *(link(U_v, x_p_nu_m_rho, sig)
                  *adj(link(U_v, x_m_rho_p_sig, nu))
                  *adj(link(U_v, x_m_rho, sig))
                  +adj(link(U_v, x_p_nu_m_rho_m_sig, sig))
                  *adj(link(U_v, x_m_rho_m_sig, nu))
                  *link(U_v, x_m_rho_m_sig, sig)
                  )*link(U_v, x_m_rho, rho));
          }
          if constexpr(term==1) {
            res = link(XY_v, x_p_mu_m_nu, nu)*adj(link(U_v, x_m_nu, mu))
                  *(link(U_v, x_m_nu, rho)
                  *(link(U_v, x_m_nu_p_rho, sig)
                  *link(U_v, x_m_nu_p_rho_p_sig, nu)
                  *adj(link(U_v, x_p_rho, sig))
                  +adj(link(U_v, x_m_nu_p_rho_m_sig, sig))
                  *link(U_v, x_m_nu_p_rho_m_sig, nu)
                  *link(U_v, x_p_rho_m_sig, sig)
                  )*adj(link(U_v, x, rho))
                  +adj(link(U_v, x_m_nu_m_rho, rho))
                  *(link(U_v, x_m_nu_m_rho, sig)
                  *link(U_v, x_m_nu_m_rho_p_sig, nu)
                  *adj(link(U_v, x_m_rho, sig))
                  +adj(link(U_v, x_m_nu_m_rho_m_sig, sig))
                  *link(U_v, x_m_nu_m_rho_m_sig, nu)
                  *link(U_v, x_m_rho_m_sig, sig)
                  )*link(U_v, x_m_rho, rho));
          }
          if constexpr(term==2) {
            U1 = adj(link(U_v, x_p_nu, mu));
            res = (link(U_v, x_p_mu, sig)
                  *adj(link(XY_v, x_p_mu_p_sig, nu))
                  *adj(link(U_v, x_p_mu_p_nu, sig))
                  +adj(link(U_v, x_p_mu_m_sig, sig))
                  *adj(link(XY_v, x_p_mu_m_sig, nu))
                  *link(U_v, x_p_mu_p_nu_m_sig, sig)
                  )*U1*link(U_v, x_p_nu, rho)*adj(link(U_v, x_p_rho, nu))*adj(link(U_v, x, rho))
                  +(link(U_v, x_p_mu, sig)
                  *adj(link(XY_v, x_p_mu_p_sig, nu))
                  *adj(link(U_v, x_p_mu_p_nu, sig))
                  +adj(link(U_v, x_p_mu_m_sig, sig))
                  *adj(link(XY_v, x_p_mu_m_sig, nu))
                  *link(U_v, x_p_mu_p_nu_m_sig, sig)
                  )*U1*adj(link(U_v, x_p_nu_m_rho, rho))*adj(link(U_v, x_m_rho, nu))*link(U_v, x_m_rho, rho);
          }
          if constexpr(term==3) {
            U1 = adj(link(U_v, x_m_nu, mu));
            res = (link(U_v, x_p_mu, sig)
                  *link(XY_v, x_p_mu_m_nu_p_sig, nu)
                  *adj(link(U_v, x_p_mu_m_nu, sig))
                  +adj(link(U_v, x_p_mu_m_sig, sig))
                  *link(XY_v, x_p_mu_m_nu_m_sig, nu)
                  *link(U_v, x_p_mu_m_nu_m_sig, sig)
                  )*U1*link(U_v, x_m_nu, rho)*link(U_v, x_m_nu_p_rho, nu)*adj(link(U_v, x, rho))
                  +(link(U_v, x_p_mu, sig)
                  *link(XY_v, x_p_mu_m_nu_p_sig, nu)
                  *adj(link(U_v, x_p_mu_m_nu, sig))
                  +adj(link(U_v, x_p_mu_m_sig, sig))
                  *link(XY_v, x_p_mu_m_nu_m_sig, nu)
                  *link(U_v, x_p_mu_m_nu_m_sig, sig)
                  )*U1*adj(link(U_v, x_m_nu_m_rho, rho))*link(U_v, x_m_nu_m_rho, nu)*link(U_v, x_m_rho, rho);
          }
          if constexpr(term==4) {
            res = (link(U_v, x_p_mu, rho)
                  *(link(U_v, x_p_mu_p_rho, sig)
                  *adj(link(XY_v, x_p_mu_p_rho_p_sig, nu))
                  *adj(link(U_v, x_p_mu_p_nu_p_rho, sig))
                  +adj(link(U_v, x_p_mu_p_rho_m_sig, sig))
                  *adj(link(XY_v, x_p_mu_p_rho_m_sig, nu))
                  *link(U_v, x_p_mu_p_nu_p_rho_m_sig, sig)
                  )*adj(link(U_v, x_p_mu_p_nu, rho))
                  +adj(link(U_v, x_p_mu_m_rho, rho))
                  *(link(U_v, x_p_mu_m_rho, sig)
                  *adj(link(XY_v, x_p_mu_m_rho_p_sig, nu))
                  *adj(link(U_v, x_p_mu_p_nu_m_rho, sig))
                  +adj(link(U_v, x_p_mu_m_rho_m_sig, sig))
                  *adj(link(XY_v, x_p_mu_m_rho_m_sig, nu))
                  *link(U_v, x_p_mu_p_nu_m_rho_m_sig, sig)
                  )*link(U_v, x_p_mu_p_nu_m_rho, rho)
                  )*adj(link(U_v, x_p_nu, mu))*adj(link(U_v, x, nu));
          }
          if constexpr(term==5) {
            res = (link(U_v, x_p_mu, rho)
                  *(link(U_v, x_p_mu_p_rho, sig)
                  *link(XY_v, x_p_mu_m_nu_p_rho_p_sig, nu)
                  *adj(link(U_v, x_p_mu_m_nu_p_rho, sig))
                  +adj(link(U_v, x_p_mu_p_rho_m_sig, sig))
                  *link(XY_v, x_p_mu_m_nu_p_rho_m_sig, nu)
                  *link(U_v, x_p_mu_m_nu_p_rho_m_sig, sig)
                  )*adj(link(U_v, x_p_mu_m_nu, rho))
                  +adj(link(U_v, x_p_mu_m_rho, rho))
                  *(link(U_v, x_p_mu_m_rho, sig)
                  *link(XY_v, x_p_mu_m_nu_m_rho_p_sig, nu)
                  *adj(link(U_v, x_p_mu_m_nu_m_rho, sig))
                  +adj(link(U_v, x_p_mu_m_rho_m_sig, sig))
                  *link(XY_v, x_p_mu_m_nu_m_rho_m_sig, nu)
                  *link(U_v, x_p_mu_m_nu_m_rho_m_sig, sig)
                  )*link(U_v, x_p_mu_m_nu_m_rho, rho)
                  )*adj(link(U_v, x_m_nu, mu))*link(U_v, x_m_nu, nu);
          }
          if constexpr(term==6) {
            res = link(U_v, x_p_mu, nu)
                  *(link(U_v, x_p_mu_p_nu, rho)
                  *(link(U_v, x_p_mu_p_nu_p_rho, sig)
                  *link(XY_v, x_p_nu_p_rho_p_sig, mu)
                  *adj(link(U_v, x_p_nu_p_rho, sig))
                  +adj(link(U_v, x_p_mu_p_nu_p_rho_m_sig, sig))
                  *link(XY_v, x_p_nu_p_rho_m_sig, mu)
                  *link(U_v, x_p_nu_p_rho_m_sig, sig)
                  )*adj(link(U_v, x_p_nu, rho))
                  +adj(link(U_v, x_p_mu_p_nu_m_rho, rho))
                  *(adj(link(U_v, x_p_mu_p_nu_m_rho_m_sig, sig))
                  *link(XY_v, x_p_nu_m_rho_m_sig, mu)
                  *link(U_v, x_p_nu_m_rho_m_sig, sig)
                  +link(U_v, x_p_mu_p_nu_m_rho, sig)
                  *link(XY_v, x_p_nu_m_rho_p_sig, mu)
                  *adj(link(U_v, x_p_nu_m_rho, sig))
                  )*link(U_v, x_p_nu_m_rho, rho)
                  )*adj(link(U_v, x, nu));
          }
          if constexpr(term==7) {
            res = adj(link(U_v, x_p_mu_m_nu, nu))
                  *(link(U_v, x_p_mu_m_nu, rho)
                  *(link(U_v, x_p_mu_m_nu_p_rho, sig)
                  *link(XY_v, x_m_nu_p_rho_p_sig, mu)
                  *adj(link(U_v, x_m_nu_p_rho, sig))
                  +adj(link(U_v, x_p_mu_m_nu_p_rho_m_sig, sig))
                  *link(XY_v, x_m_nu_p_rho_m_sig, mu)
                  *link(U_v, x_m_nu_p_rho_m_sig, sig)
                  )*adj(link(U_v, x_m_nu, rho))
                  +adj(link(U_v, x_p_mu_m_nu_m_rho, rho))
                  *(link(U_v, x_p_mu_m_nu_m_rho, sig)
                  *link(XY_v, x_m_nu_m_rho_p_sig, mu)
                  *adj(link(U_v, x_m_nu_m_rho, sig))
                  +adj(link(U_v, x_p_mu_m_nu_m_rho_m_sig, sig))
                  *link(XY_v, x_m_nu_m_rho_m_sig, mu)
                  *link(U_v, x_m_nu_m_rho_m_sig, sig)
                  )*link(U_v, x_m_nu_m_rho, rho)
                  )*link(U_v, x_m_nu, nu);
          }
          if constexpr(term==8) {
            res = link(U_v, x_p_mu, nu)*adj(link(U_v, x_p_nu, mu))
                  *(link(U_v, x_p_nu, rho)
                  *(link(U_v, x_p_nu_p_rho, sig)
                  *link(XY_v, x_p_rho_p_sig, nu)
                  *adj(link(U_v, x_p_rho, sig))
                  +adj(link(U_v, x_p_nu_p_rho_m_sig, sig))
                  *link(XY_v, x_p_rho_m_sig, nu)
                  *link(U_v, x_p_rho_m_sig, sig)
                  )*adj(link(U_v, x, rho))
                  +adj(link(U_v, x_p_nu_m_rho, rho))
                  *(link(U_v, x_p_nu_m_rho, sig)
                  *link(XY_v, x_m_rho_p_sig, nu)
                  *adj(link(U_v, x_m_rho, sig))
                  +adj(link(U_v, x_p_nu_m_rho_m_sig, sig))
                  *link(XY_v, x_m_rho_m_sig, nu)
                  *link(U_v, x_m_rho_m_sig, sig)
                  )*link(U_v, x_m_rho, rho));
          }
          if constexpr(term==9) {
            res = adj(link(U_v, x_p_mu_m_nu, nu))*adj(link(U_v, x_m_nu, mu))
                  *(link(U_v, x_m_nu, rho)
                  *(link(U_v, x_m_nu_p_rho, sig)
                  *adj(link(XY_v, x_m_nu_p_rho_p_sig, nu))
                  *adj(link(U_v, x_p_rho, sig))
                  +adj(link(U_v, x_m_nu_p_rho_m_sig, sig))
                  *adj(link(XY_v, x_m_nu_p_rho_m_sig, nu))
                  *link(U_v, x_p_rho_m_sig, sig)
                  )*adj(link(U_v, x, rho))
                  +adj(link(U_v, x_m_nu_m_rho, rho))
                  *(link(U_v, x_m_nu_m_rho, sig)
                  *adj(link(XY_v, x_m_nu_m_rho_p_sig, nu))
                  *adj(link(U_v, x_m_rho, sig))
                  +adj(link(U_v, x_m_nu_m_rho_m_sig, sig))
                  *adj(link(XY_v, x_m_nu_m_rho_m_sig, nu))
                  *link(U_v, x_m_rho_m_sig, sig)
                  )*link(U_v, x_m_rho, rho));
          }
          if constexpr(term==10) {
            U1 = adj(link(U_v, x_p_nu, mu));
            res = (link(U_v, x_p_mu, sig)
                  *link(U_v, x_p_mu_p_sig, nu)
                  *adj(link(U_v, x_p_mu_p_nu, sig))
                  +adj(link(U_v, x_p_mu_m_sig, sig))
                  *link(U_v, x_p_mu_m_sig, nu)
                  *link(U_v, x_p_mu_p_nu_m_sig, sig)
                  )*U1*link(U_v, x_p_nu, rho)*link(XY_v, x_p_rho, nu)*adj(link(U_v, x, rho))
                  +(adj(link(U_v, x_p_mu_m_sig, sig))
                  *link(U_v, x_p_mu_m_sig, nu)
                  *link(U_v, x_p_mu_p_nu_m_sig, sig)
                  +link(U_v, x_p_mu, sig)
                  *link(U_v, x_p_mu_p_sig, nu)
                  *adj(link(U_v, x_p_mu_p_nu, sig))
                  )*U1*adj(link(U_v, x_p_nu_m_rho, rho))*link(XY_v, x_m_rho, nu)*link(U_v, x_m_rho, rho);
          }
          if constexpr(term==11) {
            U1 = adj(link(U_v, x_m_nu, mu));
            res = (link(U_v, x_p_mu, sig)
                  *adj(link(U_v, x_p_mu_m_nu_p_sig, nu))
                  *adj(link(U_v, x_p_mu_m_nu, sig))
                  +adj(link(U_v, x_p_mu_m_sig, sig))
                  *adj(link(U_v, x_p_mu_m_nu_m_sig, nu))
                  *link(U_v, x_p_mu_m_nu_m_sig, sig)
                  )*U1*link(U_v, x_m_nu, rho)*adj(link(XY_v, x_m_nu_p_rho, nu))*adj(link(U_v, x, rho))
                  +(link(U_v, x_p_mu, sig)
                  *adj(link(U_v, x_p_mu_m_nu_p_sig, nu))
                  *adj(link(U_v, x_p_mu_m_nu, sig))
                  +adj(link(U_v, x_p_mu_m_sig, sig))
                  *adj(link(U_v, x_p_mu_m_nu_m_sig, nu))
                  *link(U_v, x_p_mu_m_nu_m_sig, sig)
                  )*U1*adj(link(U_v, x_m_nu_m_rho, rho))*adj(link(XY_v, x_m_nu_m_rho, nu))*link(U_v, x_m_rho, rho);
          }
          if constexpr(term==12) {
            res = (link(U_v, x_p_mu, rho)
                  *(link(U_v, x_p_mu_p_rho, sig)
                  *link(U_v, x_p_mu_p_rho_p_sig, nu)
                  *adj(link(U_v, x_p_mu_p_nu_p_rho, sig))
                  +adj(link(U_v, x_p_mu_p_rho_m_sig, sig))
                  *link(U_v, x_p_mu_p_rho_m_sig, nu)
                  *link(U_v, x_p_mu_p_nu_p_rho_m_sig, sig)
                  )*adj(link(U_v, x_p_mu_p_nu, rho))
                  +adj(link(U_v, x_p_mu_m_rho, rho))
                  *(link(U_v, x_p_mu_m_rho, sig)
                  *link(U_v, x_p_mu_m_rho_p_sig, nu)
                  *adj(link(U_v, x_p_mu_p_nu_m_rho, sig))
                  +adj(link(U_v, x_p_mu_m_rho_m_sig, sig))
                  *link(U_v, x_p_mu_m_rho_m_sig, nu)
                  *link(U_v, x_p_mu_p_nu_m_rho_m_sig, sig)
                  )*link(U_v, x_p_mu_p_nu_m_rho, rho)
                  )*adj(link(U_v, x_p_nu, mu))*link(XY_v, x, nu);
          }
          if constexpr(term==13) {
            res = (adj(link(U_v, x_p_mu_m_rho, rho))
                  *(link(U_v, x_p_mu_m_rho, sig)
                  *adj(link(U_v, x_p_mu_m_nu_m_rho_p_sig, nu))
                  *adj(link(U_v, x_p_mu_m_nu_m_rho, sig))
                  +adj(link(U_v, x_p_mu_m_rho_m_sig, sig))
                  *adj(link(U_v, x_p_mu_m_nu_m_rho_m_sig, nu))
                  *link(U_v, x_p_mu_m_nu_m_rho_m_sig, sig)
                  )*link(U_v, x_p_mu_m_nu_m_rho, rho)
                  +link(U_v, x_p_mu, rho)
                  *(adj(link(U_v, x_p_mu_p_rho_m_sig, sig))
                  *adj(link(U_v, x_p_mu_m_nu_p_rho_m_sig, nu))
                  *link(U_v, x_p_mu_m_nu_p_rho_m_sig, sig)
                  +link(U_v, x_p_mu_p_rho, sig)
                  *adj(link(U_v, x_p_mu_m_nu_p_rho_p_sig, nu))
                  *adj(link(U_v, x_p_mu_m_nu_p_rho, sig))
                  )*adj(link(U_v, x_p_mu_m_nu, rho))
                  )*adj(link(U_v, x_m_nu, mu))*adj(link(XY_v, x_m_nu, nu));
          }
          coalescedWrite(F_v[x->_offset](mu), F_v(x->_offset)(mu) + c7*res);
  } } } } )
}

// 
// Lepage and Naik stencil kernels
// 

inline std::vector<Coordinate> createHISQLepageSmearStencil() {
  std::vector<Coordinate> shifts;
  for (int mu = 0; mu < Nd; mu++) {
    for (int nu = 0; nu < Nd; nu++) {
      hisqAppendShift(shifts, shiftSignal::NO_SHIFT);   // 0: x
      hisqAppendShift(shifts, nu);                      // 1: x+nu
      hisqAppendShift(shifts, nu, nu);                  // 2: x+2nu
      hisqAppendShift(shifts, Back(nu));                // 3: x-nu
      hisqAppendShift(shifts, Back(nu), Back(nu));      // 4: x-2nu
      hisqAppendShift(shifts, mu);                      // 5: x+mu
      hisqAppendShift(shifts, mu, nu);                  // 6: x+mu+nu
      hisqAppendShift(shifts, mu, Back(nu));            // 7: x+mu-nu
      hisqAppendShift(shifts, mu, Back(nu), Back(nu));  // 8: x+mu-2nu
    }
  }
  return shifts;
}

accelerator_inline int hisqStencilIndexLepageSmear(int mu, int nu) {
  return 9*(nu + Nd*mu);
}

template<class GF>
void hisqLepageSmear(
  GF& U_fat,
  GF& U,
  GeneralLocalStencil& gStencilLP,
  int mu,
  RealD clp
) {
  autoView(U_v    , U    , AcceleratorRead);
  autoView(U_fat_v, U_fat, AcceleratorWrite);
  auto gLP_v = gStencilLP.View(AcceleratorRead);
  int Nsites = U_v.size();
  int Nsimd  = U.Grid()->Nsimd();

  typedef decltype(link(U_v, gLP_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Nsimd, {
    U3matrix res;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;
      int s = hisqStencilIndexLepageSmear(mu, nu);

      auto x       = gLP_v.GetEntry(s+0, site);  // x
      auto x_p_nu  = gLP_v.GetEntry(s+1, site);  // x+nu
      auto x_p_2nu = gLP_v.GetEntry(s+2, site);  // x+2nu
      auto x_m_nu  = gLP_v.GetEntry(s+3, site);  // x-nu
      auto x_m_2nu = gLP_v.GetEntry(s+4, site);  // x-2nu
      auto x_p_mu  = gLP_v.GetEntry(s+5, site);  // x+mu
      auto x_p_mu_p_nu  = gLP_v.GetEntry(s+6, site);  // x+mu+nu
      auto x_p_mu_m_nu  = gLP_v.GetEntry(s+7, site);  // x+mu-nu
      auto x_p_mu_m_2nu = gLP_v.GetEntry(s+8, site);  // x+mu-2nu

      res = link(U_v, x, nu)
           *link(U_v, x_p_nu, nu)
           *link(U_v, x_p_2nu, mu)
           *adj(link(U_v, x_p_mu_p_nu, nu))
           *adj(link(U_v, x_p_mu, nu))
          + adj(link(U_v, x_m_nu, nu))
           *adj(link(U_v, x_m_2nu, nu))
           *link(U_v, x_m_2nu, mu)
           *link(U_v, x_p_mu_m_2nu, nu)
           *link(U_v, x_p_mu_m_nu, nu);

      coalescedWrite(U_fat_v[x->_offset](mu), U_fat_v(x->_offset)(mu) + clp*res);
  } } )
}

template<class GF>
void hisqLepageStaple(
  GF& U_fat,
  GF& U_3link,
  GF& U,
  GeneralLocalStencil& gStencil,
  int mu,
  RealD clp
) {
  autoView(U_v      , U      , AcceleratorRead);
  autoView(U_fat_v  , U_fat  , AcceleratorWrite);
  autoView(U_3link_v, U_3link, AcceleratorRead);
  auto gStencil_v = gStencil.View(AcceleratorRead);
  int Nsites = U_v.size();
  int Nsimd  = U.Grid()->Nsimd();

  typedef decltype(link(U_v, gStencil_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Nsimd, {
    U3matrix res;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;
      int s = hisqStencilIndex3(mu, nu);

      auto x_p_mu      = gStencil_v.GetEntry(s+0, site);
      auto x_p_nu      = gStencil_v.GetEntry(s+1, site);
      auto x           = gStencil_v.GetEntry(s+2, site);
      auto x_p_mu_m_nu = gStencil_v.GetEntry(s+3, site);
      auto x_m_nu      = gStencil_v.GetEntry(s+4, site);

      res = link(U_v, x, nu)*link(U_3link_v, x_p_nu, nu)*adj(link(U_v, x_p_mu, nu))
          + adj(link(U_v, x_m_nu, nu))*link(U_3link_v, x_m_nu, nu)*link(U_v, x_p_mu_m_nu, nu);

      coalescedWrite(U_fat_v[x->_offset](mu), U_fat_v(x->_offset)(mu) + clp*res);
  } } )
}

inline std::vector<Coordinate> createHISQLepageStencil() {
  std::vector<Coordinate> shifts;
  for (int mu = 0; mu < Nd; mu++)
  for (int nu = 0; nu < Nd; nu++) {
    hisqAppendShift(shifts, Back(mu), Back(nu));       // 0:  -mu-nu
    hisqAppendShift(shifts, Back(mu));                  // 1:  -mu
    hisqAppendShift(shifts, Back(mu), nu);              // 2:  -mu+nu
    hisqAppendShift(shifts, Back(nu), Back(nu));        // 3:  -2nu
    hisqAppendShift(shifts, Back(nu));                  // 4:  -nu
    hisqAppendShift(shifts, shiftSignal::NO_SHIFT);     // 5:  0
    hisqAppendShift(shifts, nu);                        // 6:  +nu
    hisqAppendShift(shifts, nu, nu);                    // 7:  +2nu
    hisqAppendShift(shifts, mu, Back(nu), Back(nu));    // 8:  +mu-2nu
    hisqAppendShift(shifts, mu, Back(nu));              // 9:  +mu-nu
    hisqAppendShift(shifts, mu);                        // 10: +mu
    hisqAppendShift(shifts, mu, nu);                    // 11: +mu+nu
    hisqAppendShift(shifts, mu, mu, Back(nu));          // 12: +2mu-nu
    hisqAppendShift(shifts, mu, mu);                    // 13: +2mu
  }
  return shifts;
}

// Stencil index for Lepage derivative stencil (14 entries per (mu, nu) pair)
accelerator_inline int hisqStencilIndexLepage(int mu, int nu) {
  return 14*(nu + Nd*mu);
}

template<int term, class GF>
void hisqLepageDeriv(
  GF& Fghost,
  GF& Ughost,
  GF& XYghost,
  GeneralLocalStencil& gStencilLP,
  RealD clp,
  int mu
) {
  autoView(U_v , Ughost , AcceleratorRead);
  autoView(XY_v, XYghost, AcceleratorRead);
  autoView(F_v , Fghost , AcceleratorWrite);
  int Nsites = U_v.size();
  auto gLP_v = gStencilLP.View(AcceleratorRead);

  typedef decltype(link(U_v, gLP_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Fghost.Grid()->Nsimd(), {
    U3matrix res;
    for (int nu = 0; nu < Nd; nu++) {
      if (nu == mu) continue;

      int s = hisqStencilIndexLepage(mu, nu);

      auto x_m_mu_m_nu  = gLP_v.GetEntry(s+0 , site);
      auto x_m_mu       = gLP_v.GetEntry(s+1 , site);
      auto x_m_mu_p_nu  = gLP_v.GetEntry(s+2 , site);
      auto x_m_2nu      = gLP_v.GetEntry(s+3 , site);
      auto x_m_nu       = gLP_v.GetEntry(s+4 , site);
      auto x            = gLP_v.GetEntry(s+5 , site);
      auto x_p_nu       = gLP_v.GetEntry(s+6 , site);
      auto x_p_2nu      = gLP_v.GetEntry(s+7 , site);
      auto x_p_mu_m_2nu = gLP_v.GetEntry(s+8 , site);
      auto x_p_mu_m_nu  = gLP_v.GetEntry(s+9 , site);
      auto x_p_mu       = gLP_v.GetEntry(s+10, site);
      auto x_p_mu_p_nu  = gLP_v.GetEntry(s+11, site);
      auto x_p_2mu_m_nu = gLP_v.GetEntry(s+12, site);
      auto x_p_2mu      = gLP_v.GetEntry(s+13, site);

      res = Zero();

      if constexpr(term==0) {
        res += link(U_v, x, mu) * adj(link(XY_v, x_p_mu, nu)) * adj(link(U_v, x_p_mu, nu));
        res += link(U_v, x, mu) * link(XY_v, x_p_mu_m_nu, nu) * link(U_v, x_p_mu_m_nu, nu);
        res += link(U_v, x, nu) * link(XY_v, x, nu) * link(U_v, x, mu);
        res += adj(link(U_v, x_m_nu, nu)) * adj(link(XY_v, x_m_nu, nu)) * link(U_v, x, mu);
        res += link(U_v, x, mu) * link(U_v, x_p_mu, nu) * link(XY_v, x_p_mu, nu);
        res += link(U_v, x, mu) * adj(link(U_v, x_p_mu_m_nu, nu)) * adj(link(XY_v, x_p_mu_m_nu, nu));
        res += adj(link(XY_v, x, nu)) * adj(link(U_v, x, nu)) * link(U_v, x, mu);
        res += link(XY_v, x_m_nu, nu) * link(U_v, x_m_nu, nu) * link(U_v, x, mu);
      }

      if constexpr(term==1) {
        res += adj(link(U_v, x_m_mu, mu)) * adj(link(XY_v, x_m_mu, nu))
             * link(U_v, x_m_mu_p_nu, mu) * link(U_v, x_p_nu, mu) * adj(link(U_v, x_p_mu, nu));
        res += adj(link(U_v, x_m_mu, mu)) * link(XY_v, x_m_mu_m_nu, nu)
             * link(U_v, x_m_mu_m_nu, mu) * link(U_v, x_m_nu, mu) * link(U_v, x_p_mu_m_nu, nu);
        res += link(U_v, x, nu) * link(U_v, x_p_nu, mu) * link(U_v, x_p_mu_p_nu, mu)
             * link(XY_v, x_p_2mu, nu) * adj(link(U_v, x_p_mu, mu));
        res += adj(link(U_v, x_m_nu, nu)) * link(U_v, x_m_nu, mu) * link(U_v, x_p_mu_m_nu, mu)
             * adj(link(XY_v, x_p_2mu_m_nu, nu)) * adj(link(U_v, x_p_mu, mu));
        res += link(U_v, x, nu) * link(U_v, x_p_nu, nu) * adj(link(XY_v, x_p_2nu, mu))
             * adj(link(U_v, x_p_mu_p_nu, nu)) * adj(link(U_v, x_p_mu, nu));
      }

      if constexpr(term==2) {
        res += adj(link(U_v, x_m_nu, nu)) * adj(link(U_v, x_m_2nu, nu)) * adj(link(XY_v, x_m_2nu, mu))
             * link(U_v, x_p_mu_m_2nu, nu) * link(U_v, x_p_mu_m_nu, nu);
        res += adj(link(U_v, x_m_mu, mu)) * link(U_v, x_m_mu, nu)
             * link(U_v, x_m_mu_p_nu, mu) * link(U_v, x_p_nu, mu) * link(XY_v, x_p_mu, nu);
        res += adj(link(U_v, x_m_mu, mu)) * adj(link(U_v, x_m_mu_m_nu, nu))
             * link(U_v, x_m_mu_m_nu, mu) * link(U_v, x_m_nu, mu) * adj(link(XY_v, x_p_mu_m_nu, nu));
        res += adj(link(XY_v, x, nu)) * link(U_v, x_p_nu, mu) * link(U_v, x_p_mu_p_nu, mu)
             * adj(link(U_v, x_p_2mu, nu)) * adj(link(U_v, x_p_mu, mu));
        res += link(XY_v, x_m_nu, nu) * link(U_v, x_m_nu, mu) * link(U_v, x_p_mu_m_nu, mu)
             * link(U_v, x_p_2mu_m_nu, nu) * adj(link(U_v, x_p_mu, mu));
      }

      coalescedWrite(F_v[x->_offset](mu), F_v(x->_offset)(mu) + clp*adj(res));
  } } )
}

inline std::vector<Coordinate> createHISQNaikStencil() {
  std::vector<Coordinate> shifts;
  for (int mu = 0; mu < Nd; mu++) {
    hisqAppendShift(shifts, shiftSignal::NO_SHIFT); // 0: x (identity)
    hisqAppendShift(shifts, mu);                    // 1: x+mu
    hisqAppendShift(shifts, mu, mu);                // 2: x+2mu
    hisqAppendShift(shifts, Back(mu));              // 3: x-mu
    hisqAppendShift(shifts, Back(mu), Back(mu));    // 4: x-2mu
  }
  return shifts;
}

// Stencil index for Naik stencil (5 entries per mu)
accelerator_inline int hisqStencilIndexNaik(int mu) {
  return 5*mu;
}

template<class GF>
void hisqNaikSmear(
  GF& W_naik,
  GF& U,
  GeneralLocalStencil& gStencil,
  int mu,
  RealD cnaik
) {
  autoView(U_v    , U     , AcceleratorRead);
  autoView(W_v    , W_naik, AcceleratorWrite);
  auto gStencil_v = gStencil.View(AcceleratorRead);
  int Nsites = U_v.size();
  int Nsimd  = U.Grid()->Nsimd();

  typedef decltype(link(U_v, gStencil_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Nsimd, {
    int s = hisqStencilIndexNaik(mu);
    auto x       = gStencil_v.GetEntry(s+0, site);
    auto x_p_mu  = gStencil_v.GetEntry(s+1, site);
    auto x_p_2mu = gStencil_v.GetEntry(s+2, site);

    coalescedWrite(W_v[site](mu), 
      cnaik*link(U_v, x, mu)*link(U_v, x_p_mu, mu)*link(U_v, x_p_2mu, mu));
  } )
}

template<class GF>
void hisqNaikDeriv(
  GF& Fghost,
  GF& Ughost,
  GF& XYghost,
  GeneralLocalStencil& gStencilNaik,
  RealD cnaik,
  int mu
) {
  autoView(U_v , Ughost , AcceleratorRead);
  autoView(XY_v, XYghost, AcceleratorRead);
  autoView(F_v , Fghost , AcceleratorWrite);
  int Nsites = U_v.size();
  auto gStencil_v = gStencilNaik.View(AcceleratorRead);

  typedef decltype(link(U_v, gStencil_v.GetEntry(0,0), 0)) U3matrix;

  accelerator_for(site, Nsites, Fghost.Grid()->Nsimd(), {
    int s = hisqStencilIndexNaik(mu);
    auto x       = gStencil_v.GetEntry(s+0, site);
    auto x_p_mu  = gStencil_v.GetEntry(s+1, site);
    auto x_p_2mu = gStencil_v.GetEntry(s+2, site);
    auto x_m_mu  = gStencil_v.GetEntry(s+3, site);
    auto x_m_2mu = gStencil_v.GetEntry(s+4, site);

    U3matrix res;
    res = link(U_v, x_p_mu, mu)*link(U_v, x_p_2mu, mu)*link(XY_v, x, mu);
    res += link(U_v, x_p_mu, mu)*link(XY_v, x_m_mu, mu)*link(U_v, x_m_mu, mu);
    res += link(XY_v, x_m_2mu, mu)*link(U_v, x_m_2mu, mu)*link(U_v, x_m_mu, mu);

    coalescedWrite(F_v[site](mu), F_v(site)(mu) + cnaik*res);
  } )
}

NAMESPACE_END(Grid);

#endif // QCD_UTILS_HISQ_STENCIL_KERNELS_H
