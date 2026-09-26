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

// forward declaration: implemented below
template<class Field> class ConfigurationBase;

template<class Field>
class Primal {
/**
 * @class Grid::Primal
 * @brief
 * @author
 * @details
 * 
 */
private: 
  const ConfigurationBase<Field>* _owner;
  std::size_t _id;
  const Field* _ptr; 

public:
  Primal(const ConfigurationBase<Field>* owner, std::size_t id, const Field& field):
    _owner(owner), _id(id), _ptr(&field) { GRID_ASSERT(!field.Grid()->_isCheckerBoarded); }
  explicit Primal(const Field& field): Primal(nullptr, 0, field) { }
  Primal(const Primal&) = default;
  Primal(Primal&&) = default;

public:
  const ConfigurationBase<Field>* owner() const { return _owner; }
  std::size_t id() const { return _id; }
  const Field& resolve() const { return *_ptr; }

public:
  Primal& operator=(const Primal&) = delete;
  Primal& operator=(Primal&&) = default;
};

template<class Field>
using Primals = std::vector<Primal<Field>>;

template<class Field>
using ActionContract = std::pair<const void*, Primals<Field>>;

template<class Field>
struct PrimalCotangentPair: public Field {
/**
 * @struct Grid::PrimalCotangentPair
 * @brief
 * @author
 * @details
 * 
 */
public:
  Primal<Field> _primal;

public:
  using Field::operator=;
  using Field::operator+=;
  using Field::operator*=;

  explicit PrimalCotangentPair(Primal<Field> p):
    Field(p.resolve().Grid()), _primal(std::move(p)) {
    this->Checkerboard() = _primal.resolve().Checkerboard();
    static_cast<Field&>(*this) = Zero();
  }

  explicit PrimalCotangentPair(const Field& field):
    PrimalCotangentPair(Primal<Field>(field)) { }

public:
  const Primal<Field>& primal() const { return _primal; }
  
  const Field& gauge() const { return _primal.resolve(); }
  
public:
  PrimalCotangentPair& operator=(const Zero&)
  { static_cast<Field&>(*this) = Zero(); return *this; }

public:
  PrimalCotangentPair& operator+=(const PrimalCotangentPair& rhs) {
    GRID_ASSERT(_primal.owner() == rhs._primal.owner() && _primal.id() == rhs._primal.id());
    static_cast<Field&>(*this) += static_cast<const Field&>(rhs);
    return *this;
  }

  PrimalCotangentPair& operator*=(RealD weight)
  { static_cast<Field&>(*this) = weight * static_cast<const Field&>(*this); return *this; }
};

template<class Field>
class PrimalCotangentPairs: public std::vector<PrimalCotangentPair<Field>> {
/**
 * @class Grid::PrimalCotangentPairs
 * @brief
 * @author
 * @details
 */
public:
  using std::vector<PrimalCotangentPair<Field>>::vector;

  explicit PrimalCotangentPairs(const Field& field)
  { this->emplace_back(field); }

  explicit PrimalCotangentPairs(const ActionContract<Field>& contract)
  { for (const auto& primal : contract.second) this->emplace_back(primal); }

public:
  PrimalCotangentPairs& operator=(const Zero&)
  { for (auto& pair : *this) pair = Zero(); return *this; }

public:
  PrimalCotangentPairs& operator+=(PrimalCotangentPairs rhs) {
    for (auto& pair : rhs) {
      auto it = std::find_if(this->begin(), this->end(), [&](const auto& p) {
        return p.primal().owner() == pair.primal().owner() && p.primal().id() == pair.primal().id();
      });
      if (it == this->end()) this->push_back(std::move(pair));
      else *it += pair;
    }
    return *this;
  }

  PrimalCotangentPairs& operator*=(RealD weight)
  { for (auto& pair : *this) pair *= weight; return *this; }
};

///////////////////////////////////
// Smart configuration base class
///////////////////////////////////
enum class ConfigurationState { NotReady, Ready, Stale };

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

/////////////////////////
// new opt-in interface
/////////////////////////
public:
  using ConfigurationState = Grid::ConfigurationState;
  
protected:
  ConfigurationState _state = ConfigurationState::NotReady;

private:
  [[noreturn]] void _hasNotOptedIn() const
  { GRID_ASSERT(0 && "ConfigurationBase subclass has not opted in to link interface"); }

  void _establish() const {
    GRID_ASSERT(
      _state != ConfigurationState::NotReady && 
      "configuration links unavailable: set_Field not called or ConfigurationBase "
      "subclass has not opted in to link interface"
    );
  }

  void _validate(const Primal<Field>& p) const
  { GRID_ASSERT(p.owner() == this && "primal belongs to another configuration"); }

protected:
  void set() { _state = ConfigurationState::Stale; }
  
  void smeared() 
  { if (_state == ConfigurationState::Ready) { _state = ConfigurationState::Stale; } }
  
  void fulfilled() { 
    _establish(); 
    if (_state == ConfigurationState::Stale) { _state = ConfigurationState::Ready; } 
  }

  void leftTrivializedCotangent(Field& UdSdU, const Field& U) const {
    for (int mu = 0; mu < Nd; ++mu)
    { pokeLorentz(UdSdU, -peekLorentz(U, mu) * peekLorentz(UdSdU, mu), mu); }
  }

public: // opting in means implementing these virtual methods
  virtual void smear() { _hasNotOptedIn(); }
  virtual void pullback(Field&, PrimalCotangentPairs<Field>&) const { _hasNotOptedIn(); }
  virtual Primal<Field> primal(std::size_t id) const { _hasNotOptedIn(); }
  virtual const Field& fundamental() const { _hasNotOptedIn(); }

public:
  PrimalCotangentPair<Field> pair(const Primal<Field>& p) const
  { _validate(p); return PrimalCotangentPair<Field>(p); }

  PrimalCotangentPairs<Field> pairs(const Primals<Field>& primals) const {
    PrimalCotangentPairs<Field> result;
    result.reserve(primals.size());
    for (auto& p : primals) result.push_back(pair(p));
    return result;
  }
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

/////////////////////////
// Opt-in link interface
/////////////////////////
private:
  bool _hasOptedIn = false;

public:
  bool hasOptedIn() const { return _hasOptedIn; }

protected:
  template <class Operator>
  ActionContract<GaugeField> signContract(Operator& op, Primals<GaugeField>& primals) {
    GRID_ASSERT(!primals.empty());
    _hasOptedIn = true;
    return ActionContract<GaugeField>(op.identity(), primals);
  }

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

/////////////////////
// Helper procedures
/////////////////////

template <int Index, class Field, class vobj>
void PokeIndex(PrimalCotangentPair<Field>& target, const Lattice<vobj>& value, int idx) {
  Field& cotangent = target;
  GridBase* grid = cotangent.Grid();
  GridBase* source = value.Grid();

  if (source->_isCheckerBoarded)
  { GRID_ASSERT(value.Checkerboard() == Even || value.Checkerboard() == Odd); }

  if (grid == source) { conformable(cotangent, value); PokeIndex<Index>(cotangent, value, idx); return; }

  GRID_ASSERT(!grid->_isCheckerBoarded && source->_isCheckerBoarded);
  GRID_ASSERT(grid->_fdimensions == source->_fdimensions);
  GRID_ASSERT(grid->_processors == source->_processors);
  GRID_ASSERT(grid->_processor_coor == source->_processor_coor);
  GRID_ASSERT(grid->_simd_layout == source->_simd_layout);

  auto component = PeekIndex<Index>(cotangent, idx);
  acceleratorSetCheckerboard(component, value, value.Grid()->_checker_dim);
  PokeIndex<Index>(cotangent, component, idx);
}

template <class Field, class vobj>
void pokeLorentz(PrimalCotangentPair<Field>& target, const Lattice<vobj>& value, int idx)
{ PokeIndex<LorentzIndex>(target, value, idx); }

/** @brief */
template<class Field>
void conformable(const Primals<Field>& fields, GridBase* grid) 
{ for (const auto& field : fields) conformable(field.resolve().Grid(), grid); }

/** @brief */
template<class Field>
void conformable(const PrimalCotangentPairs<Field>& pairs, GridBase* grid) 
{ for (const auto& pair : pairs) conformable(pair.Grid(), grid); }

/** @brief accumulates a callback's contribution on its output grid. */
template<class vobj, class Callable>
void accumulate(Lattice<vobj>& out, const ActionContract<Lattice<vobj>>&, RealD weight, Callable&& work) {
  Lattice<vobj> partial(out.Grid());

  partial.Checkerboard() = out.Checkerboard();
  partial = Zero();

  std::forward<Callable>(work)(partial);

  GridBase* grid = partial.Grid();
  if (grid == out.Grid()) {
    if (!grid->_isCheckerBoarded) { partial.Checkerboard() = out.Checkerboard(); }
    out += weight*partial;
  } else {
    GRID_ASSERT(!out.Grid()->_isCheckerBoarded && grid->_isCheckerBoarded);
    
    Lattice<vobj> previous(grid);
    int cb = partial.Checkerboard();
    int dim = grid->_checker_dim;

    acceleratorPickCheckerboard(cb, previous, out, dim);
    previous += weight*partial;
    acceleratorSetCheckerboard(out, previous, dim);
  }
}

/** @brief temporaries use the operator's primals and full gauge grids. */
template<class Field, class Callable>
void accumulate(PrimalCotangentPairs<Field>& out, const ActionContract<Field>& contract, RealD weight, Callable&& work) {
  PrimalCotangentPairs<Field> partial(contract);
  std::forward<Callable>(work)(partial);
  partial *= weight;
  out += std::move(partial);
}

/** @brief convenience overload for unit weight accumulation. */
template<class Field, class GaugeField, class Callable>
void accumulate(Field& out, const ActionContract<GaugeField>& contract, Callable&& work)
{ accumulate(out, contract, 1.0, std::forward<Callable>(work)); }

NAMESPACE_END(Grid);

#endif // ACTION_BASE_H
