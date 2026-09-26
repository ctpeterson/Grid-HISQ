/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/smearing/HISQConfiguration.h

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

/**
 * @file HISQConfiguration.h
 * @author Curtis Taylor Peterson
 */

#pragma once

#include <Grid/Grid.h>
#include <Grid/qcd/utils/HighlyImprovedStaggeredFermionImpl.h>
#include <deque>

#ifndef QCD_SMEARING_HISQ_CONFIGURATION_H
#define QCD_SMEARING_HISQ_CONFIGURATION_H

NAMESPACE_BEGIN(Grid);

inline RealD calcNaikEpsilon(RealD m) {
  /**
   * @brief Calculates Naik epsilon according to MILC prescription
   * @author Curtis Taylor Peterson
   * @details
   * According to [MILC Collaboration (2010)], the Naik epsilon correction, Naik 
   * epsilon calculated from heavy quark mass am by combining Eqn 26 and Eqn 27
   * from [Follana, E. et al.].
   * References:
   * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
   * * MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501
   */
  RealD m2 = m*m;
  return NAIKEPS1*m2 + NAIKEPS2*m2*m2 + NAIKEPS3*m2*m2*m2 + NAIKEPS4*m2*m2*m2*m2;
}

struct HISQContext {
  HISFContext _fat7;
  UnitaryProjectionContext _proj;
  HISFContext _asqtad;
  RealD _epsilon;

private:
  static HISFContext _defaultFat7() {
    return HISFContext(F7L1, F7L3, F7L5, F7L7);
  }

  static HISFContext _defaultAsqtad() {
    return HISFContext(ASQL1, ASQL3, ASQL5, ASQL7, LEPAGE, 0.0);
  }

  static UnitaryProjectionContext _defaultProjection() {
    UnitaryProjectionContext projection(
      CayleyHamiltonProjection, 
      MIMDCollaborationDerivative
    );
    projection.setBackupSVD(BACKUPSVD);
    projection.setRelativeSVDTolerance(RELBACKUPSVDTOLERANCE);
    projection.setAbsoluteSVDTolerance(ABSBACKUPSVDTOLERANCE);
    projection.setDerivativeEigenvalueCutoff(REUNITDERIVCUTOFF);
    return projection;
  }

public:
  HISQContext(RealD epsilon = 0.0): 
    HISQContext(_defaultFat7(), _defaultAsqtad(), epsilon) { }

  HISQContext(HISFContext asqtad, RealD epsilon = 0.0): 
    HISQContext(_defaultFat7(), asqtad, epsilon) { }

  HISQContext(
    HISFContext fat7,
    HISFContext asqtad,
    RealD epsilon = 0.0
  ):HISQContext(fat7, asqtad, _defaultProjection(), epsilon) { }

  HISQContext(
    HISFContext fat7,
    HISFContext asqtad,
    UnitaryProjectionContext proj,
    RealD epsilon = 0.0
  ):_fat7(fat7), _proj(proj), _asqtad(asqtad), _epsilon(epsilon) { 
    _asqtad.c0 += epsilon/8.0; 
    _fat7.naik = 0.0;
    _asqtad.naik = 0.0;
    _fat7.c1 = -_fat7.c1;
    _fat7.c3 = -_fat7.c3;
    _asqtad.c1 = -_asqtad.c1;
    _asqtad.c3 = -_asqtad.c3;
  }

  HISQContext(UnitaryProjectionContext proj, RealD epsilon = 0.0): 
    HISQContext(_defaultFat7(), _defaultAsqtad(), proj, epsilon) { }

public:
  HISFContext fat7() const { return _fat7; }
  HISFContext asqtad() const { return _asqtad; }
  UnitaryProjectionContext proj() const { return _proj; }
};

template <class Impl>
class HISQConfiguration: 
  public ConfigurationBase<typename Impl::GaugeField> {
/**
 * @class Grid::HISQConfiguration
 * @brief
 * @author
 * @details
 * 
 * Note: we explicitly reject the legacy smearing interface for safety
 */
public: INHERIT_GIMPL_TYPES(Impl);

private:
  mutable HighlyImprovedStaggeredFermionImpl<Impl> _hisf;
  HISFContext _fat7Ctx;
  UnitaryProjectionContext _projCtx;
  std::vector<HISFContext> _asqtadCtx;

private:
  GaugeField* ThinLink;               // U
  GaugeField Fat7Link;                // V
  GaugeField UnitaryFat7Link;         // W
  std::deque<GaugeField> AsqtadLinks; // X, indexed by primal ID minus one

public:
  /** @brief constructor for opt-in interface */
  HISQConfiguration(GridCartesian* grid): 
    _hisf(grid, false), 
    ThinLink(NULL), 
    Fat7Link(grid),
    UnitaryFat7Link(grid) { }

  HISQConfiguration(const HISQConfiguration&) = delete;

public:
  HISQConfiguration& operator=(const HISQConfiguration&) = delete;

private:
  void _validateReadiness() const {
    GRID_ASSERT(this->_state == ConfigurationState::Ready);
    GRID_ASSERT(ThinLink != NULL);
  }

public: // provide promises: bound to operators by Action's signContract methods
  /**
   * @brief Create an action's primals during HMC application setup
   * @author Curtis Taylor Peterson
   * @details
   * @code
   * HISQConfiguration<HMCWrapper::ImplPolicy> policy(GridPtr);
   * action.signContract(policy.promise(context));
   * TheHMC.Run(policy);
   * @endcode
   */
  Primals<GaugeField> promise(const HISQContext& ctx) {
    if (!_asqtadCtx.empty()) {
      GRID_ASSERT(_fat7Ctx == ctx.fat7() && "fat7 mismatch");
      GRID_ASSERT(_projCtx == ctx.proj() && "projection mismatch");
    }

    const std::size_t id = AsqtadLinks.size() + 1;
    AsqtadLinks.emplace_back(_hisf.ugrid);
    try { _asqtadCtx.push_back(ctx.asqtad()); }
    catch (...) { AsqtadLinks.pop_back(); throw; }
    if (id == 1) { _fat7Ctx = ctx.fat7(); _projCtx = ctx.proj(); }
    
    this->smeared();

    return {primal(0), primal(id)};
  }

  /** @brief Register default smearing and projection with the requested epsilon */
  Primals<GaugeField> promise(RealD epsilon = 0.0) 
  { return promise(HISQContext(epsilon)); }

  /** @brief Register default smearing and projection with the requested asqtad */
  Primals<GaugeField> promise(const HISFContext& asqtad)
  { return promise(HISQContext(asqtad)); }

public: // implement opt-in interface
  Primal<GaugeField> primal(std::size_t id) const {
    GRID_ASSERT(!AsqtadLinks.empty() && "no HISQ context registered");
    if (id == 0) { return {this, id, UnitaryFat7Link}; }
    GRID_ASSERT(id <= AsqtadLinks.size() && "unregistered HISQ output");
    return {this, id, AsqtadLinks[id - 1]};
  }

  const GaugeField& fundamental() const { _validateReadiness(); return *ThinLink; }

  void smear() {
    this->smeared();
    GRID_ASSERT(ThinLink != NULL && "HISQ links not prepared; call set_Field");
    GRID_ASSERT(!AsqtadLinks.empty() && "no HISQ context registered");

    double begin = usecond();
    
    _hisf.smear(Fat7Link, *ThinLink, _fat7Ctx);
    _hisf.project(UnitaryFat7Link, Fat7Link, _projCtx);
    for (std::size_t id = 0; id < AsqtadLinks.size(); ++id)
    { _hisf.smear(AsqtadLinks[id], UnitaryFat7Link, _asqtadCtx[id]); }

    std::cout << GridLogMessage 
              << "Smearing in " 
              << (usecond() - begin) 
              << " ms" 
              << std::endl; 
    this->fulfilled();
  }

  void pullback(GaugeField& dSdU, PrimalCotangentPairs<GaugeField>& inputs) const {
    conformable(dSdU.Grid(), Fat7Link.Grid());
    _validateReadiness();
    for (const auto& input : inputs) {
      const auto& primal = input.primal();
      GRID_ASSERT(primal.owner() == this && "owner mismatch");
      GRID_ASSERT(primal.id() <= AsqtadLinks.size());
      conformable(input, Fat7Link);
    }

    double start = usecond();
    GaugeField tmp(dSdU.Grid());
    GaugeField dSdW(dSdU.Grid());

    { // asqtad pullback
      dSdW = Zero();

      for (const auto& input : inputs) {
        const GaugeField& deriv = input;
        const std::size_t id = input.primal().id();

        if (id != 0) {
          _hisf.smearDerivative(tmp, adj(deriv), UnitaryFat7Link, _asqtadCtx[id - 1]);
          dSdW += tmp;
        } else { dSdW += adj(deriv); }
    } }

    { // unitary fat7 pullback
      UnitaryProjection<Impl> projection(_projCtx);
    
      if (_projCtx.derivativeMethod == JinOsbornDerivative)
      { projection.derivative(tmp, dSdW, UnitaryFat7Link, Fat7Link); }
      else { projection.derivative(tmp, dSdW, Fat7Link); }
      _hisf.smearDerivative(dSdU, tmp, *ThinLink, _fat7Ctx);
      dSdU = adj(dSdU);
    }
    
    this->leftTrivializedCotangent(dSdU, *ThinLink);

    std::cout << GridLogMessage 
              << "GaugeConfiguration: Smeared Force chain rule took " 
              << (usecond() - start) 
              << " ms" 
              << std::endl; 
  };

public: // implement legacy interface
  void set_Field(GaugeField& U) { this->set(); ThinLink = &U; smear(); }

  GaugeField& get_U(bool smeared = false) { 
    _validateReadiness();
    if (smeared) { return AsqtadLinks[0]; }
    else { return *ThinLink; }
  }

  GaugeField& get_SmearedU() { return get_U(true); }

  void smeared_force(GaugeField& UdSdU) { // legacy path only --- rejected
    GRID_ASSERT(0 && 
      "HISQConfiguration does not support legacy smeared_force. "
      "HISQ forces (using the ImprovedStaggeredFermion operator) "
      "require separate one-hop and Naik input derivatives "
      "and joint configuration pullback, which is not supported by "
      "Grid's legacy smearing interface. Enable the action's opt-in "
      "link interface with signContract(); see relevant "
      "documentation and code samples."
    );
  }

};

NAMESPACE_END(Grid);

#endif // QCD_SMEARING_HISQ_CONFIGURATION_H
