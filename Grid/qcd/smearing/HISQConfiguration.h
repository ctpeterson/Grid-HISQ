/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/smearing/HISQConfiguration.h
Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

Copyright (C) 2023 

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

NAMESPACE_BEGIN(Grid);

template <class Gimpl, class SmearingImpl>
class HISQConfiguration: public ConfigurationBase<typename Gimpl::Field>
{
public: INHERIT_GIMPL_TYPES(Gimpl);

private:
  SmearingImpl& hisq;
  GaugeField* ThinLinks;
  GaugeField V, W, X, WWW;

public:
  HISQConfiguration(
    GridCartesian* grid,
    SmearingImpl& _hisq
  ):hisq(_hisq), 
    ThinLinks(NULL), 
    V(&grid), 
    W(&grid), 
    X(&grid), 
    WWW(&grid) { }

public:
  virtual bool hasLongLink() const { return true; }

  virtual void fill_smearedSet(GaugeField& U) {
    ThinLinks = &U;
    GaugeField R(U.Grid());

    if (ThinLinks == NULL) 
    { std::cout << GridLogError << "[HISQConfiguration] Error in ThinLinks pointer\n"; return; }

    hisq.rephase(R, U); // phase in
    
    hisq.smear(V, R);
    hisq.project(W, V);
    hisq.smear(X, WWW, W);

    hisq.rephase(X, X);     // <-+
    hisq.rephase(WWW, WWW); // <-+- phase out
  }

  virtual void set_Field(GaugeField& U) { 
    double start = usecond();
    fill_smearedSet(U); 
    double end = usecond();
    double time = (end - start) / 1e3;
    std::cout << GridLogMessage << "HISQ smearing in " << time << " ms" << std::endl;
  }

  virtual GaugeField& get_SmearedU() { return X; }

  virtual GaugeField& get_SmearedLongU() { return WWW; }

  virtual GaugeField& get_U(bool smeared = false) 
  { if (smeared) { return X; } else { return *ThinLinks; } }

  virtual void smeared_multilink_force(
    GaugeField& UdSdU, 
    const GaugeField& dSdX,  // <-+- should *not* be pre-multiplied by link
    const GaugeField& dSdWWW // <-+
  ) {
    double start = usecond();

    GaugeField R(UdSdU.Grid());
    GaugeField force(UdSdU.Grid());

    hisq.rephase(R, *ThinLinks); // <-+
    hisq.rephase(X, X);          // <-|
    hisq.rephase(WWW, WWW);      // <-+- phase in

    hisq.smearDerivative(force, dSdX, dSdWWW, W);
    hisq.projectionDerivative(force, force, V);
    hisq.smearDerivative(UdSdU, force, R);

    hisq.rephase(X, X);     // <-+
    hisq.rephase(WWW, WWW); // <-+- phase out

    for (int mu = 0; mu < Nd; ++mu) {
      GaugeLinkField tmp(UdSdU.Grid());
      tmp = peekLorentz(R, mu) * peekLorentz(UdSdU, mu);
      pokeLorentz(UdSdU, tmp, mu);
    }

    double end = usecond();
    double time = (end - start) / 1e3;
    std::cout << GridLogMessage << "HISQ derivative in " << time << " ms" << std::endl;
  }
};

NAMESPACE_END(Grid);