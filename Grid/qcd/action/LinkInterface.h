/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/action/LinkInterface.h
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
 * @file LinkInterface.h
 * @brief User interface that establishes contract between Configuration container 
 * objects and FermionOperator objects
 * @details
 * This (fairly beautiful, imo) interface defines a shared vocabulary through which 
 * FermionOperator, Action, and configuration container objects communicate the manner 
 * in which operators receive gauge fields and how derivatives with respect to those 
 * fields are pulled back to the fundamental field.
 * 
 * This is an opt-in interface that serves to refine Grid's traditional approach
 * to managing link information for HMC. An action explicitly selects this path;
 * a configuration exposing the interface does not activate it on its own.
 *
 * The interface is composed of three core components:
 * - LinkInputs: Ordered, borrowed fields supplied to an operator
 * - LinkDerivatives: Owned derivatives with respect to those inputs
 * - LinkMap: Access to configuration outputs and their joint pullback
 *
 * The action's shared helper resolves input selections, imports each distinct
 * operator, and collects complete derivatives from the action's numerical core.
 * The configuration then propagates the collected contributions through its own
 * field constructions. Operators retain responsibility for their internal link
 * constructions; actions retain responsibility for solves, signs, and weights.
 *
 * The result of the configuration pullback is a raw derivative. Multiplication
 * by the fundamental links belongs to force assembly, and projection onto the
 * momentum Lie algebra belongs to the integrator. Neither operation is performed
 * by the types in this header.
 * 
 * the geometry of hybrid Monte Carlo
 * ----------------------------------
 * 
 * !!!! TODO !!!!
 */

#pragma once

#ifndef LINKINTERFACE_H
#define LINKINTERFACE_H

NAMESPACE_BEGIN(Grid);

template <class Field>
class LinkInputs {
/**
 * @class Grid::LinkInputs
 * @brief Ordered sequence of field references
 * @author Curtis Taylor Peterson
 * @details
 * This is a sequence of borrowed, read-only field references supplied to a
 * FermionOperator object. Fields are ordered according to the ordinary
 * ImportGauge overload that the sequence is intended to serve. Each argument
 * position of that overload is referred to as a "port".
 *
 * The same input sequence is supplied for import and differentiation. Its
 * interpretation belongs to the operator; this container carries neither
 * configuration handles nor assumptions about an operator's preferred arity.
 * Two ports may refer to the same field without becoming one port.
 *
 * The container owns its pointer sequence, but does not own or copy the fields.
 * Referenced fields must remain alive throughout its use. Construction rejects
 * empty sequences and null pointers; indexing is bounds-checked. Read-only access
 * here does not prevent the field's owner from changing it, so callers must also
 * keep the input state consistent between import and differentiation.
 */
private:
  std::vector<const Field*> _fields;

public:
  explicit LinkInputs(std::vector<const Field*> fields): _fields(std::move(fields)) {
    GRID_ASSERT(!_fields.empty() && "empty input sequence");
    for (auto f : _fields) { GRID_ASSERT(f != nullptr && "null input field"); }
  }

public:
  std::size_t size() const { return _fields.size(); }
  const Field& operator[](std::size_t index) const { return *_fields.at(index); }

public:
  void conformable(GridBase* grid) const 
  { for (auto field : _fields) { GRID_ASSERT(field->Grid() == grid); } }
};

template <class Field>
class LinkDerivatives {
/**
 * @class Grid::LinkDerivatives
 * @brief Owning collection of complete raw derivatives
 * @author Curtis Taylor Peterson
 * @details
 * An owning collection of derivative fields, one for each port of the selected
 * ImportGauge overload, in the same order. FermionOperator objects write complete
 * results into LinkDerivatives objects; the action's numerical core supplies the
 * signs and weights used to accumulate them. The shared helper associates each
 * port with its configuration handle before requesting a joint pullback.
 *
 * "Complete" means that the operator has accounted for all of its dependence on
 * an input, including internal link constructions and adjointed occurrences. The 
 * fields have neither been multiplied by the input links nor projected onto the 
 * Lie algebra.
 *
 * Ports remain distinct even when they refer to the same input field. For
 * example, ImportGauge(U, U) has two partial derivatives, while ImportGauge(U)
 * has one complete derivative with respect to its single argument. Combining
 * contributions to the same configuration output is part of the joint pullback.
 *
 * An empty collection can serve as an accumulator: its first contribution
 * supplies the fields and their layouts. Later contributions must have matching
 * counts, grid instances, and checkerboards. These checks establish storage
 * compatibility; the caller is still responsible for matching operator and port
 * meanings. Field storage is owned here, but the associated grids must outlive it.
 *
 * Collection arithmetic follows the usual notation:
 * @code
 * auto ld = LinkDerivatives<Field>::fromInputs(inputs);
 * ld = Zero();
 * ld += weight * partial;
 * ld *= weight;
 * @endcode
 * An lvalue operand is copied for scalar multiplication. An rvalue can donate its
 * storage; an empty destination can then retain that storage during addition.
 */
private:
  std::vector<Field> _fields;

public:
  /** @brief construct an empty collection with no prescribed field layouts */
  LinkDerivatives() = default;
  
  /** @brief take ownership of the supplied derivative fields */
  explicit LinkDerivatives(std::vector<Field> fields): _fields(std::move(fields)) { }

public:
  /**
   * @brief Construct zero derivative fields using the input layouts
   * @details
   * Each field is allocated on its corresponding input's grid and receives that
   * input's checkerboard assignment. Input values are not copied, and this method
   * performs no differentiation. Repeated inputs still receive separate storage.
   * This is an allocation convenience for operators using those layouts; an
   * operator may instead supply its own collection of complete derivative fields.
   */
  static LinkDerivatives fromInputs(const LinkInputs<Field>& inputs) {
    std::vector<Field> fields;
    fields.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      fields.emplace_back(inputs[i].Grid());
      fields.back().Checkerboard() = inputs[i].Checkerboard();
      fields.back() = Zero();
    }
    return LinkDerivatives(std::move(fields));
  }

public:
  std::size_t size() const { return _fields.size(); }
  Field& operator[](std::size_t index) { return _fields.at(index); }
  const Field& operator[](std::size_t index) const { return _fields.at(index); }

public:
  /** @brief zero existing fields while retaining their layouts and storage */
  LinkDerivatives& operator=(const Zero&) 
  { for (auto& f : _fields) { f = Zero(); } return *this; }

  /** @brief add corresponding fields, copying the first contribution if empty */
  LinkDerivatives& operator+=(const LinkDerivatives& other) {
    if (_fields.empty()) { _fields = other._fields; return *this;}
    else { GRID_ASSERT(size() == other.size() && "derivative count mismatch"); }
    for (std::size_t i = 0; i < size(); ++i) 
    { conformable(_fields[i], other[i]); _fields[i] += other[i]; }
    return *this;
  }

  /** @brief add corresponding fields, taking first contribution's storage if empty */
  LinkDerivatives& operator+=(LinkDerivatives&& other) {
    if (this != &other && _fields.empty()) 
    { _fields = std::move(other._fields); return *this; }
    return *this += static_cast<const LinkDerivatives&>(other);
  }

  /** @brief scale all derivative fields in place */
  LinkDerivatives& operator*=(RealD weight) 
  { for (auto& f : _fields) { f = weight*f; } return *this; }

  /** @brief return a scaled collection, leaving borrowed operands unchanged */
  friend LinkDerivatives operator*(RealD weight, LinkDerivatives value) 
  { value *= weight; return value; }

  /** @brief scalar multiplication with the collection on the left */
  friend LinkDerivatives operator*(LinkDerivatives value, RealD weight) 
  { value *= weight; return value; }

private:
  template <class vobj>
  static void validateIndexLayout(const Field& field, const Lattice<vobj>& value) {
    GridBase* target = field.Grid();
    GridBase* source = value.Grid();
    
    if (source->_isCheckerBoarded) 
    { GRID_ASSERT(value.Checkerboard() == Even || value.Checkerboard() == Odd); }
    if (target == source) { conformable(field, value); return; }
    
    GRID_ASSERT(!target->_isCheckerBoarded && source->_isCheckerBoarded);
    GRID_ASSERT(target->_fdimensions == source->_fdimensions);
    GRID_ASSERT(target->_processors == source->_processors);
    GRID_ASSERT(target->_processor_coor == source->_processor_coor);
    GRID_ASSERT(target->_simd_layout == source->_simd_layout);
  }

public:
  /**
   * @brief Overwrite a tensor component at the sites represented by value
   * @details
   * Index is Grid's tensor index (for example LorentzIndex), idx selects its
   * component, and port selects the input derivative. The source must use the
   * destination grid or a compatible red-black grid. Its Checkerboard() selects
   * the sites to update; the opposite parity and other components are preserved.
   * The destination retains its grid and checkerboard metadata.
   */
  template <int Index, class vobj>
  void pokeIndex(const Lattice<vobj>& value, int idx, std::size_t port) {
    Field& field = _fields.at(port);
    
    validateIndexLayout(field, value);
    
    if (field.Grid() == value.Grid()) { PokeIndex<Index>(field, value, idx); } 
    else {
      auto component = PeekIndex<Index>(field, idx);
      acceleratorSetCheckerboard(component, value, value.Grid()->_checker_dim);
      PokeIndex<Index>(field, component, idx);
    }
  }

  /** @brief Add a tensor component, with the same site selection as pokeIndex */
  template <int Index, class vobj>
  void addIndex(const Lattice<vobj>& value, int idx, std::size_t port) {
    Field& field = _fields.at(port);
    auto component = PeekIndex<Index>(field, idx);

    validateIndexLayout(field, value);
    
    if (field.Grid() != value.Grid()) {
      Lattice<vobj> partial(value.Grid());
      int cb = value.Checkerboard();
      int dim = value.Grid()->_checker_dim;

      acceleratorPickCheckerboard(cb, partial, component, dim);
      component = std::move(partial);
    }
    
    component += value;
    pokeIndex<Index>(component, idx, port);
  }
};

template <class Field>
class LinkMap {
/**
 * @class Grid::LinkMap
 * @brief Two-way exposure interface for configuration links
 * @author Curtis Taylor Peterson
 * @details
 * Interface through which a configuration container exposes its available fields
 * and pulls back derivatives to the fundamental field. The concrete container
 * owns the knowledge of how its outputs are constructed and related.
 * 
 * Defines two structs:
 * - Handle: Reference to a particular output of a particular configuration
 * - Contribution: Borrowed (derivative, configuration output) pair
 * 
 * Imposes the following interface requirements on derived classes:
 * - resolve: Returns the field identified by a configuration handle.
 * - fundamental: Returns the fundamental gauge field itself.
 * - defaultBinding: Supplies the configuration's documented input selection for
 *   a given FermionOperator type, in ordinary ImportGauge argument order.
 * - pullback: Combines contributions and propagates them through the container's
 *   field constructions, writing a complete raw derivative with respect to the
 *   fundamental field.
 * 
 * Copying and moving this interface are disabled, preserving its address as the
 * identity used by handles. Handles borrow that identity rather than extending
 * the configuration's lifetime. Their output identifiers must remain stable for
 * as long as a caller retains the corresponding selections.
 */
public:
  /** @brief borrowed configuration owner and its local output identifier */
  struct Handle { const LinkMap* owner; unsigned output; };
  
  /** @brief borrowed raw derivative paired with the output it differentiates */
  struct Contribution { Handle handle; const Field* derivative; };

public:
  LinkMap() = default;
  LinkMap(const LinkMap&) = delete;
  
public:
  LinkMap& operator=(const LinkMap&) = delete;

public: // public-facing virtual methods
  virtual ~LinkMap() = default;
  
  /** @brief validate the handle's owner and output, then return the selected field */
  virtual const Field& resolve(Handle) const = 0;
  
  /** @brief return fundamental field with respect to which pullback differentiates */
  virtual const Field& fundamental() const = 0;
  
  /** @brief return documented input handles for this operator type, or report unsupported */
  virtual std::vector<Handle> defaultBinding(
    const std::type_info& operatorType
  ) const = 0; 

  /**
   * @brief Write the joint raw derivative with respect to the fundamental field
   * @details
   * The action's helper supplies its accumulated contributions together so that
   * shared dependencies can be handled in one pullback. Multiple contributions
   * may target the same output; outputs without a direct contribution can still
   * receive derivatives through dependent outputs.
   *
   * The directions argument contains borrowed derivative fields, not forward
   * perturbations. Their pointers must be nonnull and remain valid throughout
   * this synchronous call. The concrete implementation validates their handles
   * and layouts, consumes them without modifying or retaining them, and overwrites
   * result with the combined raw derivative. An empty list contributes zero.
   * Force assembly subsequently applies Fmu = -Umu Gmu with Gmu the Wirtinger 
   * pulled-back derivative calculated by this method, and the integrator performs 
   * projection onto the momentum Lie algebra.
   */
  virtual void pullback(
    Field& result,
    const std::vector<Contribution>& directions
  ) const = 0;
};

/** @brief public name for a configuration output handle */
template <class Field>
using LinkHandle = typename LinkMap<Field>::Handle;

/** @brief public name for a raw derivative and its configuration output handle */
template <class Field>
using LinkContribution = typename LinkMap<Field>::Contribution;

/** 
 * @brief Ordered configuration handles matching an operator's ImportGauge ports 
 * @details 
 * Public alias for an ordered sequence of handles selecting the fields supplied to 
 * ImportGauge ports. Explicit selections take precedence over defaults in the 
 * action's helper. A configuration must report unsupported operator types when it 
 * has no documented default; the caller can then provide an explicit binding. 
 * Exposing defaults does not itself opt an action in.
 */
template <class Field>
using LinkBinding = std::vector<LinkHandle<Field>>;

NAMESPACE_END(Grid);

#endif // LINKINTERFACE_H
