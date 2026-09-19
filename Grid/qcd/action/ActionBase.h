/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/ActionBase.h

Copyright (C) 2015-2016

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: neo <cossu@post.kek.jp>
Author: Guido Cossu <guido.cossu@ed.ac.uk>
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

#ifndef ACTION_BASE_H
#define ACTION_BASE_H

NAMESPACE_BEGIN(Grid);

///////////////////////////////////
// Smart configuration base class
///////////////////////////////////
template< class Field >
class ConfigurationBase
{
public:
  ConfigurationBase() {}
  virtual ~ConfigurationBase() {}

  ///////////////////////////////
  // Standard interface
  ///////////////////////////////
  virtual void set_Field(Field& U) = 0;
  virtual void smeared_force(Field&) = 0;
  virtual Field& get_SmearedU() = 0;
  virtual Field& get_U(bool smeared = false) = 0;

  ///////////////////////////////
  // Opt-in link interface
  ///////////////////////////////
  /**
   * @brief Exposes the configuration container's LinkMap, if supported
   * @details
   * Returns a borrowed interface for resolving configuration outputs and pulling
   * their derivatives back to the fundamental field. The default returns nullptr
   * to indicate that the configuration does not provide this interface.
   *
   * Exposing a map does not activate an action's use of it; the action must opt in
   * through useLinkMap or bindLinks, like Action's is_smeared. Ownership remains 
   * with the configuration, which must keep the map alive and at a stable address 
   * while its handles are in use.
   */
  virtual LinkMap<Field>* linkMap() { return nullptr; }
};

template <class GaugeField >
class Action 
{
public:
  bool is_smeared = false;
  RealD deriv_norm_sum;
  RealD deriv_max_sum;
  RealD Fdt_norm_sum;
  RealD Fdt_max_sum;
  int   deriv_num;
  RealD deriv_us;
  RealD S_us;
  RealD refresh_us;
  void  reset_timer(void)        {
    deriv_us = S_us = refresh_us = 0.0;
    deriv_norm_sum = deriv_max_sum=0.0;
    Fdt_max_sum =  Fdt_norm_sum = 0.0;
    deriv_num=0;
  }
  void  deriv_log(RealD nrm, RealD max,RealD Fdt_nrm,RealD Fdt_max) {
    if ( max > deriv_max_sum ) {
      deriv_max_sum=max;
    }
    deriv_norm_sum+=nrm;
    if ( Fdt_max > Fdt_max_sum ) {
      Fdt_max_sum=Fdt_max;
    }
    Fdt_norm_sum+=Fdt_nrm; deriv_num++;
  }
  RealD deriv_max_average(void)       { return deriv_max_sum; };
  RealD deriv_norm_average(void)      { return deriv_norm_sum/deriv_num; };
  RealD Fdt_max_average(void)         { return Fdt_max_sum; };
  RealD Fdt_norm_average(void)        { return Fdt_norm_sum/deriv_num; };
  RealD deriv_timer(void)        { return deriv_us; };
  RealD S_timer(void)            { return S_us; };
  RealD refresh_timer(void)      { return refresh_us; };
  void deriv_timer_start(void)   { deriv_us-=usecond(); }
  void deriv_timer_stop(void)    { deriv_us+=usecond(); }
  void refresh_timer_start(void) { refresh_us-=usecond(); }
  void refresh_timer_stop(void)  { refresh_us+=usecond(); }
  void S_timer_start(void)       { S_us-=usecond(); }
  void S_timer_stop(void)        { S_us+=usecond(); }
  /////////////////////////////
  // Heatbath?
  /////////////////////////////
  virtual void refresh(const GaugeField& U, GridSerialRNG &sRNG, GridParallelRNG& pRNG) = 0; // refresh pseudofermions
  virtual RealD S(const GaugeField& U) = 0;                             // evaluate the action
  virtual RealD Sinitial(const GaugeField& U) { return this->S(U); } ;  // if the refresh computes the action, can cache it. Alternately refreshAndAction() ?
  virtual void deriv(const GaugeField& U, GaugeField& dSdU) = 0;        // evaluate the action derivative
 
  /////////////////////////////////////////////////////////////
  // virtual smeared interface through configuration container
  /////////////////////////////////////////////////////////////
  virtual void refresh(ConfigurationBase<GaugeField> & U, GridSerialRNG &sRNG, GridParallelRNG& pRNG)
  {
    refresh(U.get_U(is_smeared),sRNG,pRNG);
  }
  virtual RealD S(ConfigurationBase<GaugeField>& U)
  {
    return S(U.get_U(is_smeared));
  }
  virtual RealD Sinitial(ConfigurationBase<GaugeField>& U) 
  {
    return Sinitial(U.get_U(is_smeared));
  }
  virtual void deriv(ConfigurationBase<GaugeField>& U, GaugeField& dSdU)
  {
    deriv(U.get_U(is_smeared),dSdU); 
    if ( is_smeared ) {
      U.smeared_force(dSdU);
    }
  }

  ///////////////////////////////
  // Logging
  ///////////////////////////////
  virtual std::string action_name()    = 0; // return the action name
  virtual std::string LogParameters()  = 0; // prints action parameters
  virtual ~Action(){}

  ///////////////////////////////
  // Opt-in link interface
  ///////////////////////////////
  /**
   * @brief Requests action evaluation through the configuration container's LinkMap
   * @details
   * An adopting action activates this path for subsequent configuration-based
   * refresh, S, Sinitial, and deriv calls. Explicit bindings remain in effect;
   * operators without an explicit binding use the map's documented defaults.
   * The configuration must provide a map and a supported default for each such
   * operator. The is_smeared flag does not select inputs on this path.
   *
   * This is a setup operation; field resolution and import occur during action
   * evaluation. Derived actions supply the activation and evaluation machinery.
   * The base implementation rejects unsupported actions through GRID_ASSERT.
   */
  virtual void useLinkMap() 
  { GRID_ASSERT(0 && "Action subclass does not support opt-in link interface"); }

  /**
   * @brief Associates an operator's input ports with configuration output handles
   * @details
   * The identity is the const void* returned by that operator's linkIdentity(),
   * identifying its canonical FermionOperator base subobject. It is compared
   * with this action's borrowed operators and is never dereferenced. The binding is
   * a nonempty sequence of handles in the argument order of the intended ordinary
   * ImportGauge overload. Repeated handles are allowed; their input ports remain
   * distinct. At evaluation, every handle must belong to the configuration's map.
   *
   * An adopting action retains the selection, replacing any previous binding for
   * that operator, and activates map-based evaluation. Explicit bindings take
   * precedence over defaults; other operators continue to use their own bindings
   * or the map's defaults. Selection does not import fields or transfer ownership
   * of the operator or configuration. Both must remain alive and their identities
   * stable while the selection is in use.
   *
   * Derived actions supply selection storage and validation. The base
   * implementation rejects unsupported actions through GRID_ASSERT.
   */
  virtual void bindLinks(const void*, const LinkBinding<GaugeField>&)
  { GRID_ASSERT(0 && "Action subclass does not support opt-in link interface"); }

  /**
   * @brief Binds an operator's inputs using its canonical link identity
   * @details
   * Obtains op.linkIdentity() and forwards to the virtual identity-based overload,
   * so the derived action still handles the selection. The member template itself
   * is non-virtual. Derived actions overriding bindLinks should publicly expose
   * this convenience overload with:
   * @code
   * using Action<GaugeField>::bindLinks;
   * @endcode
   */
  template <class Operator>
  void bindLinks(Operator& op, const LinkBinding<GaugeField>& binding) 
  { bindLinks(op.linkIdentity(), binding); }

  /** @brief convenience overload for inferring binding to internal operators */
  template <class Operator>
  void bindLinks(const LinkBinding<GaugeField>& binding) 
  { GRID_ASSERT(0 && "Action subclass does not provide automatic link binding"); }
  
};

template <class GaugeField >
class EmptyAction : public Action <GaugeField>
{
  using Action<GaugeField>::refresh;
  using Action<GaugeField>::Sinitial;
  using Action<GaugeField>::deriv;

  virtual void refresh(const GaugeField& U, GridSerialRNG &sRNG, GridParallelRNG& pRNG) { GRID_ASSERT(0);}; // refresh pseudofermions
  virtual RealD S(const GaugeField& U) { return 0.0;};                             // evaluate the action
  virtual void deriv(const GaugeField& U, GaugeField& dSdU) { GRID_ASSERT(0); };        // evaluate the action derivative

  ///////////////////////////////
  // Logging
  ///////////////////////////////
  virtual std::string action_name()    { return std::string("Level Force Log"); };
  virtual std::string LogParameters()  { return std::string("No parameters");};
};

NAMESPACE_END(Grid);

#endif // ACTION_BASE_H
