/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/action/pseudofermion/LinkCoordinator.h
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
 * @file LinkCoordinator.h
 * @brief Connective tissue that coordinates and manages the flow of link information 
 * through Configuration container objects, pseudofermion Action subclasses, and 
 * FermionOperator objects
 * @author Curtis Taylor Peterson
 * @details 
 * This header defines managerial data structures that coordinate the flow of link 
 * information through
 * 
 * forward passes: fundamental field U 
 *                 -> Configuration container (possible smearing)
 *                 -> pseudofermion Action subclasses
 *                 -> FermionOperators
 *
 * and
 * 
 * backward passes (derivatives): FermionOperators
 *                  -> pseudofermion Action subclasses
 *                  -> Configuration container (possible smearing)
 *                  -> derivative with respect to fundamental field U
 * 
 * within the context of hybrid Monte Carlo. This header is complementary to 
 * LinkInterface.h, whose responsibility is to define the user-facing external 
 * interface for connecting Configuration outputs to the inputs defined by ordinary
 * FermionOperator imports. LinkCoordinator.h implements and manages that connection
 * during pseudofermion action evaluations within Grid's HMC workflow.
 * 
 * Explicitly:
 * 
 * - Configuration owns configuration link outputs and is responsible for pulling 
 *   derivatives with respect to those outputs back to the fundamental field. The
 *   Configuration container's LinkMap exposes outputs and supplies default bindings.
 *   A binding associates those outputs, in order, with the FermionOperator's input
 *   ports, whose meanings are defined by the ordinary ImportGauge(...) overloads.
 * - FermionOperator objects import link information through their ports and prepare
 *   the internal links used in the FermionOperator's sparse matrix representation.
 *   FermionOperator is also responsible for providing complete Wirtinger derivatives 
 *   of fermion bilinears with respect to the links consumed through its input ports,
 *   including their use in internal link constructions.
 * - In-between Configuration container objects and the FermionOperator objects are
 *   pseudofermion Action subclasses that implement explicit pseudofermion actions
 *   and their derivatives. They delegate input preparation and derivative routing
 *   to the coordinator while retaining their solves, vector pairings, signs, and
 *   weights. Their numerical core remains agnostic to configuration bindings and
 *   link construction recipes.
 * 
 * Given that:
 * 
 * - LinkState holds the resolved link inputs and accumulated derivatives for one
 *   evaluation, and implements import, accumulation, and force finalization.
 * - LinkCoordinator coordinates the flow of link information between Configuration 
 *   container objects and FermionOperator objects through pseudofermion Action 
 *   subclasses via the representation of link information through the facilities 
 *   that LinkState provides and explicit forwarding of the Action object's virtual
 *   refresh, S, Sinitial, and deriv methods.
 */

#pragma once

#include <map>

#ifndef LINKCOORDINATOR_H
#define LINKCOORDINATOR_H

NAMESPACE_BEGIN(Grid);

// forward declaration of the LinkCoordinator class; implemented below
template <class Fimpl> class LinkCoordinator; 

template <class Fimpl>
class LinkState {
/**
 * @class Grid::LinkState
 * @brief Holds link inputs and accumulated derivatives for one action evaluation
 * @author Curtis Taylor Peterson
 * @details
 * This class is responsible for maintaining and transmitting the state of link 
 * information for one evaluation of a pseudofermion Action subclass's virtual
 * refresh, S, Sinitial, or deriv method. LinkCoordinator creates a fresh state
 * for each call. The state borrows operators and input fields and owns its
 * accumulated derivative fields; accumulators begin empty.
 */
public: 
  INHERIT_FIMPL_TYPES(Fimpl);
  using GaugeField = typename Fimpl::GaugeField;

public:
  friend class LinkCoordinator<Fimpl>;

private:
  struct LinkEntry {
  /**
   * @struct Grid::LinkState::LinkEntry
   * @brief Associates a FermionOperator with its inputs, binding, and derivatives
   * @author Curtis Taylor Peterson
   * @details
   * Associates a FermionOperator with its data for the current evaluation,
   * preserving the argument order of the selected ordinary import.
   * 
   * - op: borrowed pointer to the associated FermionOperator object
   * - inputs: ordered references to fields consumed by the FermionOperator's ports
   * - binding: configuration handles identifying fields associated with the
   *   FermionOperator's ports; empty for direct field calls
   * - derivatives: accumulated, weighted Wirtinger derivatives with respect to the
   *   fields passed through the FermionOperator's ports
   */
    FermionOperator<Fimpl>* op;
    LinkInputs<GaugeField> inputs;
    LinkBinding<GaugeField> binding;
    LinkDerivatives<GaugeField> derivatives;

    LinkEntry(
      FermionOperator<Fimpl>& op,
      LinkInputs<GaugeField> in,
      LinkBinding<GaugeField> bind
    ): op(&op), inputs(std::move(in)), binding(std::move(bind)) { }
  };

private:
  std::vector<LinkEntry> _entries;

public:
  /** @brief default constructor creates empty state; LinkCoordinator populates */
  LinkState() = default;
  
  /** @brief move constructor allows LinkCoordinator to return prepared state */
  LinkState(LinkState<Fimpl>&&) = default;
  
  LinkState(const LinkState<Fimpl>&) = delete;
  ~LinkState() = default;

public:
  LinkState& operator=(const LinkState&) = delete;

private:
  /** @brief finds LinkEntry belonging to a FermionOperator; asserts if absent */
  LinkEntry& entry(FermionOperator<Fimpl>& op) {
    for(auto& e : _entries) { if(e.op == &op) return e; }
    GRID_ASSERT(0 && "Operator not found in LinkState");
  }

  /** @brief calls ImportGauge with the recorded inputs for each operator */
  void import() { for(auto& e : _entries) { e.op->ImportGauge(e.inputs); } }

public:
  /** 
   * @brief Invokes Action's derivative callback and accumulates weighted result
   * @details
   * Calls callable(operator, partial, inputs) synchronously. The callback must write
   * one complete Wirtinger derivative per input argument. After checking the count, 
   * adds weight * partial to this state's accumulator for the operator. The callback
   * must not retain references to its arguments, and the borrowed state must not
   * escape the evaluation.
   */
  template <class Callable>
  void accumulate(
    FermionOperator<Fimpl>& op,
    RealD weight,
    Callable&& callable
  ) {
    LinkEntry& e = entry(op);
    LinkDerivatives<GaugeField> partial;

    std::forward<Callable>(callable)(*e.op, partial, e.inputs);
    if (partial.size() != e.inputs.size())
    { GRID_ASSERT(0 && "wrong argument partial count"); }
    
    e.derivatives += weight*std::move(partial);
  }

private:
  /** 
   * @brief Assembles Wirtinger derivatives for the direct field route
   * @details Sums the accumulated Wirtinger derivatives with respect to the shared
   * field u into d, then writes the action-facing force as an unprojected
   * left-trivialized derivative.
   */
  void assemble(GaugeField& f, const GaugeField& u) const {
    GaugeField d(u.Grid());

    d = Zero();
    for (const auto& e : _entries) { 
      for (std::size_t p = 0; p < e.derivatives.size(); ++p) 
      { conformable(d, e.derivatives[p]); d += e.derivatives[p]; }
    }

    conformable(f, u);
    for (int mu = 0; mu < Nd; ++mu) 
    { pokeLorentz(f, -peekLorentz(u, mu) * peekLorentz(d, mu), mu); }
  }

  /**
   * @brief Pulls configuration-output Wirtinger derivatives back to fundamental field
   * @details Associates accumulated Wirtinger derivatives with their
   * configuration-output handles and sends them through one joint map.pullback
   * call. With the resulting fundamental-field Wirtinger derivative d, writes the
   * action-facing force as the unprojected left-trivialized derivative.
   */
  void pullback(GaugeField& f, const LinkMap<GaugeField>& map) const {
    const GaugeField& u = map.fundamental();
    conformable(f, u);
    GaugeField d(u.Grid());
    std::vector<LinkContribution<GaugeField>> derivatives;

    for (const auto& e : _entries) {
      for (std::size_t p = 0; p < e.derivatives.size(); ++p)
      { derivatives.push_back({e.binding.at(p), &e.derivatives[p]}); }
    }

    map.pullback(d, derivatives);

    conformable(f, u);
    conformable(u, d);
    for (int mu = 0; mu < Nd; ++mu) 
    { pokeLorentz(f, -peekLorentz(u, mu) * peekLorentz(d, mu), mu); }
  }
};

template <class Fimpl>
class LinkCoordinator {
/**
 * @class Grid::LinkCoordinator
 * @brief Coordinates the flow of link information between Configuration container 
 * objects and FermionOperator objects through pseudofermion Action subclasses
 * @author Curtis Taylor Peterson
 * @details
 * Responsible for holding the persistent setup of a pseudofermion Action. Adopting 
 * actions keep one LinkCoordinator as an instance variable. It retains information 
 * regarding which operators the Action subclass borrows, whether configuration 
 * bindings are enabled, and any explicit bindings that are selected for particular 
 * operators. It then resolves inputs and prepares the borrowed FermionOperator 
 * objects for the subclass's virtual refresh, S, Sinitial, and deriv methods. Each 
 * evaluation uses a fresh LinkState; resolved inputs and derivative fields are not 
 * cached here.
 * 
 * - _useLinkMap: establishes whether Configuration-container-based evaluations should
 *   resolve bindings through the Configuration container's LinkMap
 * - _operators: the distinct FermionOperators borrowed by adopting Action subclass
 * - _bindings: explicit bindings keyed by typed operator pointer; operators absent
 *   from this map use configuration defaults when map-based evaluation is active
 *
 * Operators and configurations referenced by retained bindings must remain alive
 * at stable addresses. The const evaluation methods preserve this setup while
 * importing new fields into the borrowed operators.
 */
public:
  using GaugeField = typename Fimpl::GaugeField;

private:
  bool _useLinkMap = false;
  std::vector<FermionOperator<Fimpl>*> _operators;
  std::map<const FermionOperator<Fimpl>*, LinkBinding<GaugeField>> _bindings;

public:
  /**
   * @brief LinkCoordinator constructed from list of pointers to FermionOperators
   * @details
   * Registers borrowed operators of adopting action, rejects null pointers and empty
   * registrations, and removes duplicate pointers. Registering the same operator
   * twice leads to one import per evaluation by this coordinator. Separate actions
   * prepare their borrowed operators independently.
   */
  explicit LinkCoordinator(std::initializer_list<FermionOperator<Fimpl>*> ops) {
    for (FermionOperator<Fimpl>* op : ops) {
      if (op == nullptr) { GRID_ASSERT(0 && "null operator"); } 
    
      // repeated registration within this coordinator requires only one import
      if (std::find(_operators.begin(), _operators.end(), op) == _operators.end())
      { _operators.push_back(op); }
    }
    if (_operators.empty()) { GRID_ASSERT(0 && "Action has no operators"); }
  }

private:
  /** @brief checks that binding is nonempty and every handle has a nonnull owner */
  void validateBinding(const LinkBinding<GaugeField>& binding) const {
    if (binding.empty()) { GRID_ASSERT(0 && "empty link binding"); }
    for (LinkHandle<GaugeField> handle : binding) {
      if (handle.owner == nullptr) { GRID_ASSERT(0 && "null output owner"); }
    }
  }

  /** @brief retrieves Configuration container's LinkMap; asserts if unavailable */
  LinkMap<GaugeField>& requireLinkMap(ConfigurationBase<GaugeField>& config) const {
    LinkMap<GaugeField>* map = config.linkMap();
    if (map == nullptr) 
    { GRID_ASSERT(0 && "Action opted in, but Configuration has no link capability"); }
    return *map;
  }

  /** 
   * @brief Retrieves binding for given operator 
   * @details 
   * Returns the explicit binding if one exists; otherwise requests default binding
   * from the map for the operator's dynamic type.
   */
  LinkBinding<GaugeField> bindingFor(
    const FermionOperator<Fimpl>& op,
    const LinkMap<GaugeField>& map
  ) const {
    auto explicitBinding = _bindings.find(&op);
    if (explicitBinding != _bindings.end()) { return explicitBinding->second; }
    return map.defaultBinding(typeid(op));
  }

  /**
   * @brief Resolves link inputs from a given binding and link map
   * @details
   * Validates link binding, checks that every handle belongs to supplied LinkMap, and
   * resolves each handle into a borrowed field reference, preserving handle ordering
   * and handle repetitions.
   */
  LinkInputs<GaugeField> resolveInputs(
    const LinkBinding<GaugeField>& binding,
    const LinkMap<GaugeField>& map
  ) const {
    std::vector<const GaugeField*> fields;
    validateBinding(binding);
    for (LinkHandle<GaugeField> handle : binding) {
      if (handle.owner != &map) 
      { GRID_ASSERT(0 && "output belongs to another configuration"); }
      fields.push_back(&map.resolve(handle));
    }
    return LinkInputs<GaugeField>(std::move(fields));
  }

  /** 
   * @brief Resolves shared link state from explicit gauge field
   * @details
   * Constructs a LinkState with entries for each registered operator that is 
   * associated with the explicit gauge field as its single input. Configuration
   * bindings are empty on this route. Operators are not imported yet.
   */
  LinkState<Fimpl> resolveSharedInputs(const GaugeField& input) const {
    LinkState<Fimpl> state;
    for (FermionOperator<Fimpl>* op : _operators) {
      LinkInputs<GaugeField> inputs({&input});
      LinkBinding<GaugeField> binding;
      state._entries.emplace_back(*op, std::move(inputs), std::move(binding));
    }
    return state;
  }

  /** 
   * @brief Resolves link state from a Configuration container's LinkMap
   * @details
   * Constructs LinkState with entries for each registered operator that is associated 
   * with the link inputs resolved from the provided LinkMap. Retains each binding
   * for derivative routing. Operators are not imported yet.
   */
  LinkState<Fimpl> resolveConfigurationInputs(const LinkMap<GaugeField>& map) const {
    LinkState<Fimpl> state;
    for (FermionOperator<Fimpl>* op : _operators) {
      LinkBinding<GaugeField> binding = bindingFor(*op, map);
      LinkInputs<GaugeField> inputs = resolveInputs(binding, map);
      state._entries.emplace_back(*op, std::move(inputs), std::move(binding));
    }
    return state;
  }

public:
  /** @brief enables LinkMap resolution on subsequent Configuration evaluations */
  void useLinkMap() { _useLinkMap = true; }

  /** 
   * @brief Finds registered operator whose linkIdentity() matches supplied pointer
   * @details 
   * Validates the binding's basic structure, inserts or replaces the corresponding
   * FermionOperator's explicit selection, and enables map-based evaluation upon
   * finding a registered operator with linkIdentity() matching the supplied pointer.
   * Unrelated identities are rejected. The identity is compared only; output
   * resolution and import occur during evaluation.
   */
  void select(
    const void* identity,
    const LinkBinding<GaugeField>& binding
  ) {
    for (FermionOperator<Fimpl>* op : _operators) {
      if (op->linkIdentity() == identity) {
        validateBinding(binding);
        _bindings.insert_or_assign(op, binding);
        _useLinkMap = true;
        return;
    } }
    GRID_ASSERT(0 && "selection names an unrelated operator");
  }

public:
  /** @brief Imports operator's single input, then invokes the refresh callback */
  template <class Callable>
  void refresh(const GaugeField& u, Callable&& work) const {
    LinkState<Fimpl> state = resolveSharedInputs(u);
    state.import();
    work();
  }

  /**
   * @brief Prepares operators from Configuration, then invokes the refresh callback
   * @details 
   * Uses config.get_U(smeared) when map resolution is inactive. Otherwise resolves 
   * explicit or default bindings through LinkMap; smeared does not select inputs on 
   * that route.
   */
  template <class Callable>
  void refresh(
    ConfigurationBase<GaugeField>& config,
    bool smeared,
    Callable&& work
  ) const {
    if (!_useLinkMap) { refresh(config.get_U(smeared), work); return; }
    
    LinkMap<GaugeField>& map = requireLinkMap(config);
    LinkState<Fimpl> state = resolveConfigurationInputs(map);
    
    state.import();
    work();
  }

  /** @brief Imports each operator's single input; returns action callback's value */
  template <class Callable>
  RealD action(const GaugeField& u, Callable&& work) const {
    LinkState<Fimpl> state = resolveSharedInputs(u);
    state.import();
    return work();
  }

  /**
   * @brief Prepares operators from Configuration; returns action callback's value
   * @details 
   * Uses the same input-selection routes as configuration-based refresh. An adopting 
   * action can use this method for S or Sinitial.
   */
  template <class Callable>
  RealD action(
    ConfigurationBase<GaugeField>& config,
    bool smeared,
    Callable&& work
  ) const {
    if (!_useLinkMap) { return action(config.get_U(smeared), work); }
    
    LinkMap<GaugeField>& map = requireLinkMap(config);
    LinkState<Fimpl> state = resolveConfigurationInputs(map);
    
    state.import();
    return work();
  }

  /**
   * @brief Imports shared inputs, invokes work(state), and assembles the force
   * @details 
   * Every operator receives u as its single input, independently of stored bindings. 
   * The callback supplies weighted Wirtinger derivative contributions through 
   * LinkState's accumulate.
   */
  template <class Callable>
  void deriv(const GaugeField& u, GaugeField& force, Callable&& work) const {
    LinkState<Fimpl> state = resolveSharedInputs(u);
    
    state.import();
    work(state);
    state.assemble(force, u);
  }

  /**
   * @brief Prepares configuration inputs, invokes work(state), and finalizes force
   * @details 
   * With map resolution active, collects contributions for one joint configuration 
   * pullback. Otherwise differentiates config.get_U(smeared) through the shared-field 
   * route and calls config.smeared_force(force) if smeared is true.
   */
  template <class Callable>
  void deriv(
    ConfigurationBase<GaugeField>& config,
    GaugeField& force,
    bool smeared,
    Callable&& work
  ) const {
    if (!_useLinkMap) {
      deriv(config.get_U(smeared), force, work);
      if (smeared) { config.smeared_force(force); }
      return;
    }

    LinkMap<GaugeField>& map = requireLinkMap(config);
    LinkState<Fimpl> state = resolveConfigurationInputs(map);
    
    state.import();
    work(state);
    state.pullback(force, map);
  }
};

NAMESPACE_END(Grid);

#endif // LINKCOORDINATOR_H
