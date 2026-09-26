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
#include <cmath>
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

  void _cleanup() {
    _fat7.naik = 0.0;
    _asqtad.naik = 0.0;
    _fat7.c1 = -_fat7.c1;
    _fat7.c3 = -_fat7.c3;
    _asqtad.c1 = -_asqtad.c1;
    _asqtad.c3 = -_asqtad.c3;
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
  ):_fat7(fat7), _proj(proj), _asqtad(asqtad), _epsilon(epsilon) 
  { _asqtad.c0 += epsilon/8.0; _cleanup(); }

  HISQContext(UnitaryProjectionContext proj, RealD epsilon = 0.0): 
    HISQContext(_defaultFat7(), _defaultAsqtad(), proj, epsilon) { }

public:
  HISFContext fat7() const { return _fat7; }
  HISFContext asqtad() const { return _asqtad; }
  UnitaryProjectionContext proj() const { return _proj; }
};

// Keep identifiers 0 and 1 invalid: lower-stage links are not bindable outputs.
enum HISQLinks: unsigned { UnitaryFat7 = 2, Asqtad };

template <class Impl>
class HISQConfiguration: 
  public ConfigurationBase<typename Impl::GaugeField> {
/**
 * @class Grid::HISQConfiguration
 * @brief
 * @author
 * @details
 * 
 * Note: we explicitly reject the legacy smearing interface for safety and consistency.
 */
public:
  INHERIT_FIELD_TYPES(Impl);
  using GaugeField = typename Impl::GaugeField;
  using Derivative = PrimalCotangentPair<GaugeField>;
  using Derivatives = PrimalCotangentPairs<GaugeField>;

private:
  mutable HighlyImprovedStaggeredFermionImpl<Impl> _hisf;
  std::vector<HISQContext> _contexts;

private:
  GaugeField* ThinLink;                // U
  GaugeField Fat7Link;                 // V: shared by all contexts
  GaugeField UnitaryFat7Link;          // W: shared by all contexts
  std::deque<GaugeField> AsqtadLinks;  // X: indexed with _contexts

public:
  /** @brief constructor for opt-in interface */
  HISQConfiguration(GridCartesian* grid): 
    _hisf(grid, false), 
    ThinLink(NULL), 
    Fat7Link(grid), 
    UnitaryFat7Link(grid),
    AsqtadLinks(0, grid) { }

  HISQConfiguration(const HISQConfiguration&) = delete;
  HISQConfiguration& operator=(const HISQConfiguration&) = delete;

private:
  void _validateRegistration() const
  { GRID_ASSERT(!_contexts.empty() && "no HISQ context registered"); }

  void _validateProjection(const UnitaryProjectionContext& ctx) const {
    GRID_ASSERT(
      std::isfinite(ctx.derivativeEigenvalueCutoff) && ctx.derivativeEigenvalueCutoff >= 0.0 &&
      "invalid derivative eigenvalue cutoff"
    );
    if (ctx.derivativeMethod == MIMDCollaborationDerivative) {
      GRID_ASSERT(
        !(ctx.svdOnlyDerivative && ctx.backupSVD) &&
        "SVD-only derivative and backup SVD are incompatible"
      );
    }
    if (ctx.backupSVD) {
      GRID_ASSERT(
        ctx.projectionMethod == CayleyHamiltonProjection &&
        "backup SVD only supported for Cayley-Hamilton projection"
      );
      GRID_ASSERT(
        std::isfinite(ctx.relativeSVDTolerance) && ctx.relativeSVDTolerance >= 0.0 &&
        std::isfinite(ctx.absoluteSVDTolerance) && ctx.absoluteSVDTolerance >= 0.0 &&
        "invalid backup SVD tolerance"
      );
    }
  }

  void _validateFirstStage(const HISQContext& ctx) const {
    _validateProjection(ctx.proj());
    if (_contexts.empty()) return;
    const HISQContext& first = _contexts.front();
    GRID_ASSERT(first.fat7() == ctx.fat7() && "fat7 mismatch");
    GRID_ASSERT(first.proj() == ctx.proj() && "projection mismatch");
  }

  void _validateFirstStage() const {
    _validateRegistration();
    const HISQContext& first = _contexts.front();
    for (const HISQContext& ctx : _contexts) { _validateFirstStage(ctx); }
  }

  void _validateReady() const {
    GRID_ASSERT(
      this->_state == ConfigurationBase<GaugeField>::ConfigurationState::Ready &&
      ThinLink != NULL &&
      "HISQ links not prepared; call set_Field after registration"
    );
  }

  void _validateAsqtadLink(const std::size_t idx) const {
    _validateRegistration();
    GRID_ASSERT(!AsqtadLinks.empty() && "asqtad link not initialized"); 
    GRID_ASSERT(
      idx < _contexts.size() && idx < AsqtadLinks.size() && 
      "unregistered HISQ context"
    );
  }

  void _validateAsqtadLinks() const {
    _validateRegistration();
    GRID_ASSERT(AsqtadLinks.size() == _contexts.size() && "asqtad links not initialized");
  }

  void _validatePrimal(const Primal<GaugeField>& primal) const {
    GRID_ASSERT(primal.owner() == this && "primal does not belong to this configuration");
    GRID_ASSERT(
      primal.id() >= HISQLinks::UnitaryFat7 &&
      "HISQ primals must use UnitaryFat7 or Asqtad"
    );
    if (primal.id() >= HISQLinks::Asqtad) { _asqIdx(primal.id()); }
  }

  void _validateDerivative(const Derivative& input) const {
    _validatePrimal(input.primal());
    conformable(input, Fat7Link);
  }

  void _validateDerivatives(const Derivatives& inputs) const
  { for (const auto& input : inputs) { _validateDerivative(input); } }

private:
  HISFContext _fat7Ctx() const
  { _validateFirstStage(); return _contexts.front().fat7(); }

  HISFContext _asqtadCtx(const std::size_t idx) const
  { _validateAsqtadLink(idx); return _contexts[idx].asqtad(); }

  std::size_t _asqIdx(std::size_t output) const {
    GRID_ASSERT(output >= HISQLinks::Asqtad && "not an Asqtad output");
    const std::size_t idx = output - HISQLinks::Asqtad;
    _validateAsqtadLink(idx);
    return idx;
  }

  UnitaryProjectionContext _projCtx() const
  { _validateFirstStage(); return _contexts.front().proj(); }

private:
  void _firstLevelSmearing() {
    _hisf.smear(Fat7Link, *ThinLink, _fat7Ctx());
    _hisf.project(UnitaryFat7Link, Fat7Link, _projCtx());
  }

  void _secondLevelSmearing() {
    for (std::size_t idx = 0; idx < _contexts.size(); ++idx) 
    { _hisf.smear(AsqtadLinks[idx], UnitaryFat7Link, _asqtadCtx(idx)); }
  }

private:
  void _secondStagePullback(GaugeField& dSdW, const Derivatives& inputs) const { 
    GaugeField tmp(dSdW.Grid());
    const GaugeField& W = UnitaryFat7Link;

    dSdW = Zero();
    for (const auto& input : inputs) {
      std::size_t idx = input.primal().id();
      const GaugeField& deriv = input;

      if (idx >= HISQLinks::Asqtad) { 
        _hisf.smearDerivative(tmp, adj(deriv), UnitaryFat7Link, _asqtadCtx(_asqIdx(idx)));
        dSdW += tmp; 
      } else if (idx == HISQLinks::UnitaryFat7) { dSdW += adj(deriv); }
    }
  }

  void _firstStagePullback(GaugeField& dSdU, const GaugeField& dSdW) const {
    GaugeField dSdV(dSdU.Grid());
    const auto ctx = _projCtx();
    UnitaryProjection<Impl> projection(ctx);
    
    if (ctx.derivativeMethod == JinOsbornDerivative)
    { projection.derivative(dSdV, dSdW, UnitaryFat7Link, Fat7Link); }
    else { projection.derivative(dSdV, dSdW, Fat7Link); }
    _hisf.smearDerivative(dSdU, dSdV, *ThinLink, _fat7Ctx());
    dSdU = adj(dSdU);
  }

public:
  Primal<GaugeField> UnitaryFat7() const { return primal(HISQLinks::UnitaryFat7); }
  Primal<GaugeField> Asqtad(std::size_t idx = 0) const {
    GRID_ASSERT(idx < _contexts.size() && "unregistered HISQ context");
    return primal(HISQLinks::Asqtad + idx);
  }

public:
  /**
   * @brief Create an action's primals during HMC application setup
   * @author Curtis Taylor Peterson
   * @details
   * @code
   * HISQConfiguration<HMCWrapper::ImplPolicy> policy(GridPtr);
   * action.contract(policy.links(context));
   * TheHMC.Run(policy);
   * @endcode
   */
  Primals<GaugeField> promise(const HISQContext& ctx) {
    _validateFirstStage(ctx);

    const std::size_t index = _contexts.size();

    AsqtadLinks.emplace_back(_hisf.ugrid);
    try { _contexts.push_back(ctx); } catch (...) { AsqtadLinks.pop_back(); throw; }
    
    this->smeared();

    return {UnitaryFat7(), Asqtad(index)};
  }

  /** @brief Register default smearing and projection with the requested epsilon */
  Primals<GaugeField> promise(RealD epsilon = 0.0) { return promise(HISQContext(epsilon)); }

  /** @brief Register default smearing and projection with the requested asqtad */
  Primals<GaugeField> promise(const HISFContext& asqtad)
  { return promise(HISQContext(asqtad)); }

public: // implement opt-in link interface
  Primal<GaugeField> primal(std::size_t id) const {
    _validateAsqtadLinks();
    if (id == HISQLinks::UnitaryFat7) { return {this, id, UnitaryFat7Link}; }
    return {this, id, AsqtadLinks[_asqIdx(id)]};
  }

  const GaugeField& fundamental() const { _validateReady(); return *ThinLink; }

  void pullback(GaugeField& dSdU, Derivatives& inputs) const {
    conformable(dSdU.Grid(), Fat7Link.Grid());

    _validateReady();
    _validateAsqtadLinks();
    _validateDerivatives(inputs);

    double start = usecond();
    GaugeField dSdW(dSdU.Grid());
    
    _secondStagePullback(dSdW, inputs);
    _firstStagePullback(dSdU, dSdW);
    for (int mu = 0; mu < Nd; ++mu)
    { pokeLorentz(dSdU, -peekLorentz(*ThinLink, mu) * peekLorentz(dSdU, mu), mu); }

    std::cout << GridLogMessage 
              << "GaugeConfiguration: Smeared Force chain rule took " 
              << (usecond() - start) 
              << " ms" 
              << std::endl; 
  };

public: // implement Configuration interface 
  void set_Field(GaugeField& U) {
    this->set();
    ThinLink = &U;
    smear();
  }

  void smear() {
    this->smeared();
    GRID_ASSERT(ThinLink != NULL && "HISQ links not prepared; call set_Field");
    _validateFirstStage();
    _validateAsqtadLinks();

    double begin = usecond();
    
    _firstLevelSmearing();
    _secondLevelSmearing();

    std::cout << GridLogMessage 
              << "Smearing in " 
              << (usecond() - begin) 
              << " ms" 
              << std::endl; 
    this->fulfilled();
  }

  GaugeField& get_U(bool smeared = false) { 
    _validateReady();
    if (smeared) { _validateAsqtadLink(0); return AsqtadLinks[0]; }
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
      "link interface with contract(); see relevant "
      "documentation and code samples."
    );
  }

};

NAMESPACE_END(Grid);

#endif // QCD_SMEARING_HISQ_CONFIGURATION_H
