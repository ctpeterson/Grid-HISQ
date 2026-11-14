/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./HMC/HISQ4f.cc

Copyright (C) 2015-2016

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
#include <iomanip>
#include <regex>
#include <streambuf>
#include <limits>
#include <fstream>
#include <cstring>

using namespace Grid;
using Gimpl = PeriodicGimplD;
using GaugeField = Gimpl::GaugeField;
using Fermion = ImprovedStaggeredFermionD;
using FermionF = ImprovedStaggeredFermionF;
using FermionField = Fermion::FermionField;
using PseudoFermionActionRat = FourFlavorStaggeredEvenEvenRatioPseudoFermionAction<StaggeredImplD>;
using PseudoFermionActionReg = FourFlavorStaggeredEvenEvenPseudoFermionAction<StaggeredImplD>;
using HMCWrapper = HMCWrapperTemplate<Gimpl, InterleavedIntegrator, NoHirep, JSONReader>;

const json DEFAULT_INPUTS = {
  {"lattice-geometry", {8,8,8,16}},
  {"hmc", {
    {"trajectory-length", .1},
    {"serial-rng", "milc"},
    {"parallel-rng", "milc"},
    {"serial-seed", 123456789},
    {"parallel-seed", 123456789},
    {"gauge-start", "cold"}
  }},
  {"action", {
    {"beta", 7.5},
    {"mass", 0.005},
    {"hasenbusch-mass", 0.6},
    {"lepage", 0.0},
    {"naik", 1.0},
    {"unitary-projection", {
      {"method", "cayley-hamilton"},
      {"delta", 5e-5}
    }}
  }},
  {"gauge", {
    {"integrator", "2MN"},
    {"steps", 10}
  }},
  {"fermion", {
    {"integrator", "2MN"},
    {"steps", 10}
  }},
  {"reversibility-check", {{"frequency", 1}}},
  {"measurements", {
    {"plaquette", json::object()},
    {"polyakov", json::object()},
    {"chiral-condensate", {{"sources", 10}}}
  }}
};

// just some forward declarations
void require(bool ok, const std::string& message) { if (!ok) { throw std::runtime_error(message); } }
class QEXInput;

//
// Gauge action class
//

class HISQGaugeAction: public PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl> {
public:
  using PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl>::PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction;

  RealD S(ConfigurationBase<GaugeField>& links) override
  { links.get_SmearedU(); return S(links.get_U(false)); }

  RealD Sinitial(ConfigurationBase<GaugeField>& links) override { return S(links); }

  RealD S(const GaugeField& U) override {
    const double begin = usecond();
    const RealD action = PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl>::S(U);
    std::cout << GridLogMessage << "HISQGaugeAction: action " << (usecond()-begin)*1e-6 << std::endl;
    return action;
  }

  void deriv(const GaugeField& U, GaugeField& force) override {
    const double begin = usecond();
    PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl>::deriv(U,force);
    std::cout << GridLogMessage << "HISQGaugeAction: force " << (usecond()-begin)*1e-6 << std::endl;
  }
};

//
// HISQ solver for mixed-precision solves
//

class HISQSolver: public OperatorFunction<FermionField> {
public:
  int maxIter;
  double tol, innerTol;
  const int innerMaxIter = 50;

  ConjugateGradient<FermionField> _doubleCG;
  Fermion* _precise;
  FermionF* _single;
  std::unique_ptr<precisionChangeWorkspace> _gaugeConvert;

public:
  HISQSolver(
    double tolerance,
    int maxIterations,
    FermionF* single,
    Fermion* precise
  ): _doubleCG(tolerance,maxIterations), _precise(precise), _single(single) {
    if (_single) {
      GRID_ASSERT(_precise);
      _gaugeConvert = std::make_unique<precisionChangeWorkspace>(_single->GaugeGrid(), _precise->GaugeGrid());
    }

    tol = _doubleCG.Tolerance;
    innerTol = std::max(1e-4, tol);
    maxIter = _doubleCG.MaxIterations;
  }

private:
  void _report(const char* precision, int iterations, double residual, double begin) {
    const double seconds=(usecond()-begin)*1e-6;
    std::cout << GridLogMessage
              << "HISQSolver: " << precision
              << " iterations " << iterations
              << " residual " << residual
              << " seconds " << seconds << std::endl;
  }

public:
  void operator()(
    LinearOperatorBase<FermionField>& op,
    const FermionField& rhs,
    FermionField& solution
  ) override {
    std::cout << GridLogMessage << "HISQSolver: begin" << std::endl;
    const double begin=usecond();

    if (!_single) {
      _doubleCG(op, rhs, solution);
      _report("double", _doubleCG.IterationsToComplete, _doubleCG.TrueResidual, begin);
      return;
    }
    else { GRID_ASSERT(_precise); }

    precisionChange(_single->GetU(), _precise->GetU(), *_gaugeConvert);
    precisionChange(_single->GetUUU(), _precise->GetUUU(), *_gaugeConvert);
    _single->CopyGaugeCheckerboards(); // change so that no host/device transfer?

    SchurStaggeredOperator<FermionF, FermionF::FermionField> normal(*_single);
    MixedPrecisionConjugateGradient<FermionField, FermionF::FermionField> solve(
      tol, maxIter, innerMaxIter, _single->FermionRedBlackGrid(), normal, op
    );

    solve.InnerTolerance = innerTol;
    solve(rhs, solution);

    _report("mixed", solve.TotalInnerIterations + solve.TotalFinalStepIterations, solve.TrueResidual, begin);
  }
};

//
// deferred smearing
//

class DeferredPullback: public ConfigurationBase<HMCWrapper::Field> {
public:
  typedef HMCWrapper::Field Field;

public:
  ConfigurationBase<Field>& _links;
  mutable PrimalCotangentPairs<Field> _pending;

public:
  explicit DeferredPullback(ConfigurationBase<Field>& links): _links(links) {}

public:
  PrimalCotangentPairs<Field>& pending() { return _pending; }

public:
  Primal<Field> primal(std::size_t id) const override { return _links.primal(id); }
  const Field& fundamental() const override { return _links.fundamental(); }
  const Field resolve(std::size_t id) const override { return _links.resolve(id); }

public:
  void smear() override { _links.smear(); }
  void set_Field(Field& U) override { _links.set_Field(U); }

  Field& get_SmearedU() override { return _links.get_SmearedU(); }
  Field& get_U(bool smeared = false) override { return _links.get_U(smeared); }

public:
  void pullback(Field& UdSdU, PrimalCotangentPairs<Field>& dSdU) const override
  { _pending += std::move(dSdU); UdSdU = Zero(); }

  void smeared_force(Field& UdSdU) override { _links.smeared_force(UdSdU); }
};

//
// pseudofermion container
//

class HISQPseudoFermions: public Action<HMCWrapper::Field> {
public:
  typedef HMCWrapper::Field Field;

public:
  std::vector<Action<Field>*> _fields;

public:
  std::string action_name() override { return "HISQPseudoFermions"; }

  std::string LogParameters() override {
    std::string parameters;
    for (auto* field : _fields) { parameters += field->LogParameters(); }
    return parameters;
  }

public:
  void push_back(Action<Field>* field) {
    GRID_ASSERT(field->hasOptedIn() && "fields must use the HISQ link interface (signContract)");
    _fields.push_back(field);
  }

private:
  RealD sum(ConfigurationBase<Field>& U, bool initial) {
    RealD total = 0.0;
    for (int i = 0; i < _fields.size(); ++i) {
      const RealD term = initial ? _fields[i]->Sinitial(U) : _fields[i]->S(U);
      std::cout << GridLogMessage << action_name() << " field " << i << " S = " << term << std::endl;
      total += term;
    }
    return total;
  }

public:
  void refresh(const Field&, GridSerialRNG&, GridParallelRNG&) override
  { GRID_ASSERT(0 && "HISQPseudoFermions needs the HISQ configuration"); }
  void refresh(ConfigurationBase<Field>& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG) override
  { for (auto field=_fields.rbegin(); field!=_fields.rend(); ++field) { (*field)->refresh(U, sRNG, pRNG); } }

public:
  using Action<Field>::Sinitial;
  RealD Sinitial(ConfigurationBase<Field>& U) override { return sum(U, true); }

  RealD S(const Field&) override
  { GRID_ASSERT(0 && "HISQPseudoFermions needs the HISQ configuration"); return 0.0; }
  RealD S(ConfigurationBase<Field>& U) override { return sum(U, false); }

public:
  void deriv(const Field&, Field&) override
  { GRID_ASSERT(0 && "HISQPseudoFermions needs the HISQ configuration"); }
  void deriv(ConfigurationBase<Field>& U, Field& UdSdU) override {
    DeferredPullback deferred(U);
    Field zero(UdSdU.Grid());
    for (auto* field : _fields) { field->deriv(deferred, zero); }
    U.pullback(UdSdU, deferred.pending());
  }
};

//
// measurements
//

#if defined(RNG_MILCV6) && defined(HAVE_LIME)

// The link contract and stochastic solve follow HISQ2p1p1f. QEX uses m + D/2
// and the m*||M^-1 eta||^2 estimator with unit-variance noise.
class HISQMeasurements: public HmcObservable<HMCWrapper::Field> {
private:
  const QEXInput& _in;
  Fermion& _op;
  ActionContract<GaugeField> _contract;

public:
  HISQMeasurements(const QEXInput& in, Fermion& op, HISQConfiguration<Gimpl>& links):
    _in(in),
    _op(op),
    _contract(op.identity(), links.promise()) {}

private:
  // Split plaquette, normalized by Nc for the QEX record. Grid's plaquette
  // and Polyakov observables supply the total plaquette and the loops.
  void _gauge(const GaugeField& U) const {
    auto* grid = U.Grid();
    const RealD volume = grid->gSites();
    std::vector<Gimpl::GaugeLinkField> link(Nd, grid);
    Gimpl::ComplexField plaquette(grid);
    RealD spatial = 0, temporal = 0;

    for (int mu = 0; mu < Nd; ++mu) { link[mu] = PeekIndex<LorentzIndex>(U, mu); }

    for (int mu = 1; mu < Nd; ++mu) { for (int nu = 0; nu < mu; ++nu) {
      WilsonLoops<Gimpl>::traceDirPlaquette(plaquette, link, mu, nu);
      (mu == Tdir ? temporal : spatial) += real(TensorRemove(sum(plaquette)));
    } }

    std::cout << GridLogMessage << "Plaquette planes: " << spatial/(3*Nc*volume)
              << ' ' << temporal/(3*Nc*volume) << '\n';
  }

  void _condensate(GridParallelRNG& parallel) const;

public:
  void TrajectoryComplete(
    int traj,
    ConfigurationBase<GaugeField>& links,
    GridSerialRNG& serial,
    GridParallelRNG& parallel
  ) override;

  void TrajectoryComplete(int, GaugeField&, GridSerialRNG&, GridParallelRNG&) override
  { throw std::logic_error("HISQ measurements require a configuration"); }
};

// AddObservable owns this module, as in HISQ2p1p1f.
class HISQMeasurementModule: public HMCModuleBase<HmcObservable<HMCWrapper::Field>> {
private:
  HISQMeasurements _measurements;

public:
  template<class... Args>
  explicit HISQMeasurementModule(Args&&... args): _measurements(std::forward<Args>(args)...) {}

public:
  HmcObservable<HMCWrapper::Field>* getPtr() override { return &_measurements; }
};

#endif

//
// configuration and RNG I/O
//

#if defined(RNG_MILCV6) && defined(HAVE_LIME)

// QEX's SciDAC envelopes and MILCv6 state representation. Payload transport,
// endian conversion, checksums, and RNG evolution use Grid's implementations.
using QEXRngState = std::array<uint32_t, 12>;

static_assert(
  sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
  "QEX RNG checkpoints require IEEE binary64"
);

//
// SciDAC records
//

// Grid's close() leaves destruction of the LIME handle to its caller.
struct QEXReader: GridLimeReader {
public:
  uint64_t size;

public:
  explicit QEXReader(const std::string& file) {
    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    require(bool(stream), "Cannot open " + file);
    size = stream.tellg();
    open(file);
  }

  ~QEXReader() { limeDestroyReader(LimeR); close(); }
};

struct QEXWriter: GridLimeWriter {
public:
  QEXWriter(const std::string& file, GridBase* grid): GridLimeWriter(grid->IsBoss()) {
    int success = 1;
    if (grid->IsBoss()) { std::ofstream stream(file, std::ios::binary | std::ios::trunc); success = bool(stream); }
    grid->Broadcast(0, &success, sizeof(success));
    require(success, "Cannot create " + file);
    open(file);
  }

  ~QEXWriter() {
    if (boss_node) { limeDestroyWriter(LimeW); }
    close();
  }
};

struct QEXRecord {
public:
  QEXReader reader;
  scidacRecord record;
  uint64_t offset = 0, bytes = 0;

public:
  QEXRecord(const std::string& file, GridBase* grid, bool rngRecord = false): reader(file) {
    scidacFile meta;
    std::string xml;

    reader.readLimeObject(meta, "scidacFile", SCIDAC_PRIVATE_FILE_XML);
    require(
      meta.spacetime == Nd &&
      meta.getDimensions() == grid->GlobalDimensions().toVector() &&
      meta.volfmt == 0,
      "Expected single-file SciDAC with matching lattice geometry: " + file
    );

    // QEX RNG records omit colors/spins. Read the required fields rather than
    // requiring every member of Grid's scidacRecord.
    reader.readLimeObject(xml, SCIDAC_PRIVATE_RECORD_XML);
    XmlReader header(xml, true, "scidacRecord");

    read(header, "recordtype", record.recordtype);
    read(header, "datatype", record.datatype);
    read(header, "precision", record.precision);
    read(header, "typesize", record.typesize);
    read(header, "datacount", record.datacount);
    if (!rngRecord) { read(header, "colors", record.colors); }
    require(record.recordtype == 0, "Expected a full lattice record: " + file);

    while (limeReaderNextRecord(reader.LimeR) == LIME_SUCCESS) {
      const std::string type = limeReaderType(reader.LimeR);

      if (type == "scidac-binary-data" || type == ILDG_BINARY_DATA) {
        offset = ftello(reader.File);
        bytes = limeReaderBytes(reader.LimeR);
        break;
      }
    }

    require(
      offset > 0 && offset <= reader.size && bytes <= reader.size - offset,
      "Missing or truncated SciDAC binary record: " + file
    );
  }

public:
  void checksum(uint32_t suma, uint32_t sumb) {
    scidacChecksum checksum;
    FieldNormMetaData norm;

    reader.readScidacChecksum(checksum, norm);
    require(
      scidacChecksumVerify(checksum, suma, sumb),
      "SciDAC checksum mismatch: " + reader.filename
    );
  }
};

void writeQEXHeader(GridLimeWriter& writer, GridBase* grid, scidacRecord& record) {
  scidacFile meta(grid);
  emptyUserRecord note;

  writer.writeLimeObject(1, 0, meta, "scidacFile", SCIDAC_PRIVATE_FILE_XML);
  writer.writeLimeObject(0, 1, note, "note", SCIDAC_FILE_XML);
  writer.writeLimeObject(1, 0, record, "scidacRecord", SCIDAC_PRIVATE_RECORD_XML);
  writer.writeLimeObject(0, 0, note, "note", SCIDAC_RECORD_XML);
}

uint64_t beginQEXData(GridLimeWriter& writer, GridBase* grid, uint64_t bytes) {
  uint64_t offset = 0;
  writer.createLimeRecordHeader("scidac-binary-data", 0, 0, bytes);
  if (grid->IsBoss()) { fflush(writer.File); offset = ftello(writer.File); }
  grid->Broadcast(0, &offset, sizeof(offset));
  return offset;
}

void endQEXData(GridLimeWriter& writer, GridBase* grid, uint32_t suma, uint32_t sumb) {
  scidacChecksum checksum;
  std::ostringstream hexA, hexB;

  if (grid->IsBoss()) { fseeko(writer.File, 0, SEEK_END); limeWriterCloseRecord(writer.LimeW); }

  hexA << std::hex << suma;
  hexB << std::hex << sumb;
  checksum.suma = hexA.str();
  checksum.sumb = hexB.str();
  writer.writeLimeObject(0, 1, checksum, "scidacChecksum", SCIDAC_CHECKSUM);
}

//
// gauge fields
//

template<class V>
void readQEXGaugePayload(Lattice<V>& U, QEXRecord& file) {
  using Site = typename V::scalar_object;

  uint32_t nersc, suma, sumb;

  require(
    file.bytes == sizeof(Site)*U.Grid()->gSites(),
    "Wrong gauge record size: " + file.reader.filename
  );

  BinaryIO::readLatticeObject<V, Site>(
    U, file.reader.filename, BinarySimpleMunger<Site, Site>(),
    file.offset, getFormatString<V>(), nersc, suma, sumb
  );
  file.checksum(suma, sumb);
}

void readQEXGauge(GaugeField& U, const std::string& name) {
  QEXRecord file(name, U.Grid());
  const auto& record = file.record;

  require(
    record.datacount == Nd && record.colors == Nc &&
    (record.precision == "F" || record.precision == "D") &&
    record.typesize == 2*Nc*Nc*(record.precision == "F" ? sizeof(float) : sizeof(double)),
    "Unsupported gauge record metadata: " + name
  );

  if (record.precision == "D") { readQEXGaugePayload(U, file); }
  else {
    GridCartesian grid(
      U.Grid()->GlobalDimensions(),
      GridDefaultSimd(Nd, vComplexF::Nsimd()),
      U.Grid()->_processors
    );
    LatticeGaugeFieldF single(&grid);

    readQEXGaugePayload(single, file);
    precisionChange(U, single);
  }

  std::cout << GridLogMessage << "loading gauge file: " << name << '\n';
}

void writeQEXGauge(GaugeField& U, const std::string& name) {
  using V = GaugeField::vector_object;
  using Site = GaugeField::scalar_object;

  QEXWriter writer(name, U.Grid());
  FieldMetaData header;
  scidacFile meta;
  scidacRecord record;
  uint64_t offset;
  uint32_t nersc, suma, sumb;

  ScidacMetaData(U, header, record, meta);
  record.datatype = "QDP_D3_ColorMatrix";
  writeQEXHeader(writer, U.Grid(), record);

  // QIO requires the checksum immediately after the binary data. Grid's
  // convenience LIME writer inserts FieldNormMetaData, so write the envelope
  // here and use BinaryIO for the payload.
  offset = beginQEXData(writer, U.Grid(), sizeof(Site)*U.Grid()->gSites());
  BinaryIO::writeLatticeObject<V, Site>(
    U, name, BinarySimpleMunger<Site, Site>(),
    offset, getFormatString<V>(), nersc, suma, sumb
  );
  endQEXData(writer, U.Grid(), suma, sumb);
}

//
// MILCv6 engine and Gaussian cache state
//

QEXRngState exportQEXState(const GridRNGbase& rng, int index) {
  const auto state = rng._generators[index].state();
  const double spare = rng._gaussian[index]._spare;
  QEXRngState q{};

  std::copy_n(state.begin() + 2, 7, q.begin());
  q[7] = state[9];
  q[8] = state[11];
  q[9] = rng._gaussian[index]._hasSpare ? 0 : 1;
  std::memcpy(q.data() + 10, &spare, sizeof(spare));

  return q;
}

void importQEXState(GridRNGbase& rng, int index, const QEXRngState& q) {
  auto state = rng._generators[index].state();
  double spare;

  require(q[9] <= 1, "Invalid QEX Gaussian cache flag");
  std::memcpy(&spare, q.data() + 10, sizeof(spare));
  require(q[9] != 0 || std::isfinite(spare), "Invalid QEX cached Gaussian");

  std::copy_n(q.begin(), 7, state.begin() + 2);
  state[9] = q[7];
  state[10] = 0;
  state[11] = q[8];

  rng._generators[index].setState(state); // Grid validates the engine state.
  rng._gaussian[index]._hasSpare = q[9] == 0;
  rng._gaussian[index]._spare = spare;
}

void qexRNGCheckpoint(
  const std::string& base,
  GridSerialRNG& serial,
  GridParallelRNG& parallel,
  bool reading
) {
  auto* grid = parallel.Grid();
  const std::string name = base + ".parallelRNG";
  QEXRngState serialState = exportQEXState(serial, 0);
  std::vector<QEXRngState> parallelState(grid->lSites());
  uint32_t nersc, suma, sumb, invalid = 0;
  uint64_t offset = 0;
  int success = 1;

  // The serial file is a raw native-endian struct. The parallel file stores
  // twelve big-endian 32-bit words per site in a SciDAC envelope.
  if (grid->IsBoss()) {
    if (reading) {
      std::ifstream file(base + ".serialRNG", std::ios::binary);

      success = bool(file.read(reinterpret_cast<char*>(serialState.data()), sizeof(serialState))) &&
                file.peek() == EOF;
    } else {
      std::ofstream file(base + ".serialRNG", std::ios::binary);

      file.write(reinterpret_cast<const char*>(serialState.data()), sizeof(serialState));
      file.flush();
      success = bool(file);
    }
  }

  grid->Broadcast(0, &success, sizeof(success));
  require(success, "Missing, truncated, or unwritable serial RNG file: " + base + ".serialRNG");
  if (reading) { grid->Broadcast(0, serialState.data(), sizeof(serialState)); }

  // Read and verify the parallel payload before importing either RNG state.
  if (reading) {
    QEXRecord file(name, grid, true);
    const auto& record = file.record;

    require(
      record.datatype == "QDP_MilcRngv6" && record.precision == "F" && record.datacount == 1 &&
      record.typesize == sizeof(QEXRngState) && file.bytes == sizeof(QEXRngState)*grid->gSites(),
      "Unsupported MILCv6 RNG record metadata: " + name
    );

    offset = file.offset;
    BinaryIO::IOobject(
      uint32_t(0), grid, parallelState, name, offset, "IEEE32BIG",
      BinaryIO::BINARYIO_READ | BinaryIO::BINARYIO_LEXICOGRAPHIC, nersc, suma, sumb
    );
    file.checksum(suma, sumb);
  }

  // Gather invalid-state flags across ranks before reporting a failed import.
  try { if (reading) { importQEXState(serial, 0, serialState); } }
  catch (const std::exception&) { invalid = 1; }

  for (int site = 0; site < grid->lSites(); ++site) {
    Coordinate local;
    int index;

    grid->LocalIndexToLocalCoor(site, local);
    index = parallel.generator_idx(grid->oIndex(local), grid->iIndex(local));

    if (reading) {
      try { importQEXState(parallel, index, parallelState[site]); }
      catch (const std::exception&) { invalid = 1; }
    } else { parallelState[site] = exportQEXState(parallel, index); }
  }

  grid->GlobalSum(invalid);
  require(invalid == 0, "Invalid MILCv6 engine or Gaussian cache state in " + base);

  if (!reading) {
    QEXWriter writer(name, grid);
    scidacRecord record;

    // QEX identifies this integer-backed struct by F (32-bit words), not I.
    record.datatype = "QDP_MilcRngv6";
    record.precision = "F";
    record.typesize = sizeof(QEXRngState);
    record.datacount = 1;

    writeQEXHeader(writer, grid, record);
    offset = beginQEXData(writer, grid, sizeof(QEXRngState)*grid->gSites());
    BinaryIO::IOobject(
      uint32_t(0), grid, parallelState, name, offset, "IEEE32BIG",
      BinaryIO::BINARYIO_WRITE | BinaryIO::BINARYIO_LEXICOGRAPHIC, nersc, suma, sumb
    );
    endQEXData(writer, grid, suma, sumb);
  } else {
    std::cout << GridLogMessage << "read serial RNG file: " << base << ".serialRNG\n"
              << GridLogMessage << "read parallel RNG file: " << base << ".parallelRNG\n";
  }
}

//
// HMC checkpointer
//

class QEXCheckpointer: public BaseHmcCheckpointer<Gimpl> {
private:
  const QEXInput& _in;

public:
  explicit QEXCheckpointer(const QEXInput& in): _in(in) {}

public:
  void initialize(const CheckpointerParameters&) override {}

  void CheckpointRestore(
    int traj,
    GaugeField& U,
    GridSerialRNG& serial,
    GridParallelRNG& parallel
  ) override;

  void TrajectoryComplete(
    int traj,
    ConfigurationBase<GaugeField>& links,
    GridSerialRNG& serial,
    GridParallelRNG& parallel
  ) override;
};

struct QEXCheckpointFormat { const QEXInput& input; };

// Select the driver-local SciDAC format through Grid's metadata-templated
// loader. HMCResourceManager owns restore and post-measurement checkpointing.
namespace Grid {
template<>
class ScidacHmcCheckpointer<Gimpl, QEXCheckpointFormat>: public QEXCheckpointer {
public:
  ScidacHmcCheckpointer(const CheckpointerParameters&, const QEXCheckpointFormat& format):
    QEXCheckpointer(format.input) {}
};
}

#endif

//
// QEX input class
//

// Accept Nim's --key:value and --key=value as well as Grid's --key value.
std::string option(int argc, char** argv, const std::string& key, std::string fallback) {
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == key) { require(i+1 < argc, "Missing value for " + key); return argv[i+1]; }
    if (a.find(key + ":") == 0 || a.find(key + "=") == 0) { return a.substr(key.size()+1); }
  }
  return fallback;
}

class QEXInput {
public:
  int save = 0, reverse = 0, sources = 0;
  int serialSeed, parallelSeed, gaugeSteps, fermionSteps;
  int actionMax, forceMax, actionVerbosity, forceVerbosity;

  double beta, mass, hmass, lepage, naik, actionR2, forceR2, cutoff;

  bool plaq = false, ploop = false, mixedPrecision = false;

  std::string ensemble, start;

  Coordinate lattice, ranks, simd, simdSingle;

  json info;
  HMCparameters hmc;

private:
  void validateSimd(const Coordinate& geometry, int lanes, const std::string& key) const {
    require(geometry.size()==Nd, key+" must have four dimensions");
    int remaining = lanes;

    for (int d = 0; d < Nd; ++d) {
      require(
        remaining % geometry[d] == 0,
        key + " must match this Grid build's SIMD width (" + std::to_string(lanes) + ")"
      );

      remaining /= geometry[d];
      const int local=lattice[d]/ranks[d];

      require(
        local % geometry[d] == 0 && (local/geometry[d]) % 2 == 0,
        key + " must leave an even number of local SIMD sites in each dimension"
      );
      require(
        d != 0 || geometry[d] == 1 || (local/geometry[d]) % 4 == 0,
        key + " must leave an even SIMD extent after checkerboarding x"
      );
      require(
        ranks[d] == 1 || geometry[d] <= 2,
        key + " supports at most two SIMD lanes along a rank-split dimension"
      );
    }
    require(remaining == 1, key + " must match this Grid build's SIMD width (" + std::to_string(lanes) + ")");
  }

public:
  QEXInput(int argc, char** argv) {
    const int worldSize = GlobalSharedMemory::WorldSize;
    int requested = 1;

    const auto file = option(argc, argv, "--json", "");

    if (!file.empty()) { JSONReader reader(file); reader.readDefault("", info);}
    else { info = DEFAULT_INPUTS; }

    const auto& h = info.at("hmc");
    const auto& a = info.at("action");
    const auto& f = info.at("fermion");
    const auto& g = info.at("gauge");

    // lattice & layout geometry
    lattice = info.at("lattice-geometry").get<std::vector<int>>();
    ranks = info.value("rank-geometry", GridDefaultMpi().toVector());
    simd = info.value("simd-geometry", GridDefaultSimd(Nd, vComplexD::Nsimd()).toVector());
    simdSingle = info.value("simd-geometry-single", GridDefaultSimd(Nd, vComplexF::Nsimd()).toVector());

    // validate rank layout
    require(lattice.size() == Nd && ranks.size() == Nd, "Lattice and rank geometry must have four dimensions");
    for (int d=0; d<Nd; ++d) {
      require(lattice[d] > 0 && ranks[d] > 0, "Lattice and rank geometry entries must be positive");
      require(ranks[d] <= worldSize/requested, "rank-geometry product must match the launched MPI process count");
      require(lattice[d] % ranks[d] == 0, "rank-geometry must divide lattice-geometry in each dimension");
      requested *= ranks[d];
    }
    require(requested == worldSize, "rank-geometry product must match the launched MPI process count");

    // validate simd layout
    mixedPrecision = f.value("mixed-precision", false);
    validateSimd(simd,vComplexD::Nsimd(), "simd-geometry");
    if (mixedPrecision || info.count("simd-geometry-single"))
    { validateSimd(simdSingle,vComplexF::Nsimd(), "simd-geometry-single"); }

    ensemble = option(argc,argv,"--ensemble", "test"); // ensemble name

    // hmc information
    hmc.StartTrajectory = std::stoi(option(argc,argv, "--start", "0"));
    hmc.Trajectories = std::stoi(option(argc,argv, "--ntraj", "1"));
    hmc.NoMetropolisUntil = 0;
    hmc.PerformRandomShift = false;
    hmc.MetropolisTest = h.value("metropolis", true);

    // initialization
    start = h.at("gauge-start");
    if (start == "read") { hmc.StartingType = "CheckpointStart"; }
    else if (start == "hot" || start == "random") { hmc.StartingType = "HotStart"; }
    else if (start == "cold" || start == "frozen" || start == "symmetric") { hmc.StartingType = "ColdStart"; }
    else { throw std::runtime_error("Unknown gauge-start: " + start); }

    // RNG seeds (serial and parallel)
    serialSeed = h.at("serial-seed"); parallelSeed = h.at("parallel-seed");
    for (const auto& key : {"serial-rng","parallel-rng"}) {
      const std::string rng = h.at(key);
      require(rng == "milc" || rng == "MILC" || rng == "MilcRngv6", "Unsupported RNG: " + rng);
    }

    // bare parameters
    beta = a.at("beta");
    mass = a.at("mass");
    hmass = a.at("hasenbusch-mass");

    // HISQ improvement coefficients
    lepage = a.at("lepage");
    naik = a.at("naik");

    // HISQ force filter
    cutoff=1e-20;
    if (a.count("unitary-projection")) {
      const auto& p = a.at("unitary-projection");
      const std::string err = "This Grid HISQ driver supports cayley-hamilton projection (with SVD fallback)";
      require(p.value("method", std::string("cayley-hamilton")) == "cayley-hamilton", err);
      cutoff = p.value("delta", cutoff);
    }

    // solver parameters
    actionR2 = f.value("r2-action", 1e-20);
    forceR2 = f.value("r2-force", 1e-12);
    actionMax = f.value("maxits-action", 10000);
    forceMax = f.value("maxits-force", 10000);
    actionVerbosity = f.value("solver-verbosity-action", 1);
    forceVerbosity = f.value("solver-verbosity-force", 1);

    // global integrator parameters
    hmc.MD.trajL = h.at("trajectory-length");
    hmc.MD.name = "InterleavedIntegrator";
    hmc.MD.MDsteps = 1;

    // local integrator parameters
    require(g.at("integrator") == "2MN", "gauge.integrator must be 2MN");
    require(f.at("integrator") == "2MN", "fermion.integrator must be 2MN");
    gaugeSteps = g.at("steps");
    fermionSteps = f.at("steps");

    // checkpoint, reversibility check, and measurements
    if (info.count("checkpoint")) { save = info.at("checkpoint").at("frequency"); }
    if (info.count("reversibility-check")) { reverse = info.at("reversibility-check").at("frequency"); }
    if (info.count("measurements")) {
      const auto& m = info.at("measurements");
      plaq = m.count("plaquette");
      ploop = m.count("polyakov");
      if (m.count("chiral-condensate")) { sources=m.at("chiral-condensate").at("sources"); }
    }

    // legacy: alphas.nim had pppa bcs, despite letting the bcs be an input option
    require(a.value("boundary-conditions", std::string("pppa")) == "pppa", "only pppa boundary conditions");
  }

private:
  HISFContext _fat7(double lepage, double naik)
  { return HISFContext((1 + 3*lepage + naik)/8., -1./16, 1./64, -1./384, -lepage/16,0); }

public:
  HISQContext context() {
    auto projection = HISQContext().proj();
    projection.setDerivativeEigenvalueCutoff(cutoff);
    return HISQContext(_fat7(lepage, 0), _fat7(2.0 - lepage, naik), projection);
  }
};

//
// QEX output formatter
//

// Translate Grid's records one line at a time around Run().
class QEXOutput: public std::streambuf {
private:
  // stream and input
  std::streambuf* _saved;
  std::ostream _out;
  std::string _pending;
  const QEXInput& _in;
  double _volume;
  bool _active;

  // trajectory and solver state
  double _fermion = 0, _gauge = 0, _deltaH = 0;
  double _probability = 1, _random = 0;
  int _iterations = 0;
  double _residual = 0;
  bool _molecularDynamics = false, _solving = false, _observing = false;

  // measurement state
  bool _smearedPlaquette = false;
  double _plaquette = 0;
  ComplexD _spatialPolyakov = 0, _temporalPolyakov = 0;

  // original logging settings
  std::ios::fmtflags _flags;
  std::streamsize _precision;
  std::ios::iostate _state;
  int _messageActive, _hmcActive;

public:
  QEXOutput(const QEXInput& in, GridBase* grid):
    _saved(std::cout.rdbuf()),
    _out(_saved),
    _in(in),
    _volume(grid->gSites()),
    _active(grid->IsBoss()),
    _flags(std::cout.flags()),
    _precision(std::cout.precision()),
    _state(std::cout.rdstate()),
    _messageActive(GridLogMessage.isActive()),
    _hmcActive(GridLogHMC.isActive()) {
    _out << std::setprecision(16);
    if (!_active) { return; }

    GridLogMessage.Active(1);
    GridLogHMC.Active(1);

    std::cout.flush();
    std::cout.rdbuf(this);
    std::cout.precision(17);

    parameters();
  }

  ~QEXOutput() {
    if (!_active) { return; }
    if (!_pending.empty()) { _out << _pending << '\n'; }

    _out.flush();
    std::cout.rdbuf(_saved);
    std::cout.flags(_flags);
    std::cout.precision(_precision);
    std::cout.clear(_state);

    GridLogMessage.Active(_messageActive);
    GridLogHMC.Active(_hmcActive);
  }

  QEXOutput(const QEXOutput&) = delete;
  QEXOutput& operator=(const QEXOutput&) = delete;

public:
  void parameters();

private: // QEX records
  void _writePolyakov() {
    _out << "MEASploop spatial: " << _spatialPolyakov.real() << ' ' << _spatialPolyakov.imag()
         << " temporal: " << _temporalPolyakov.real() << ' ' << _temporalPolyakov.imag() << '\n';
  }

  void _writeSolver(double seconds) {
    if (_observing) { return; }
    if ((_molecularDynamics ? _in.forceVerbosity : _in.actionVerbosity) <= 0) { return; }

    const auto flags = _out.flags();
    const auto precision = _out.precision();
    const double gigaflops = seconds > 0 ? 1e-9*1146*_volume*_iterations/seconds : 0;

    _out << std::defaultfloat << std::showpoint << std::setprecision(4)
         << "stagSolve: " << _iterations << "  " << seconds << "s  "
         << gigaflops << "Gf/s  " << _residual*_residual << '\n';

    _out.flags(flags);
    _out.precision(precision);
  }

private: // measurement records
  void _readMeasurements(const std::string& text) {
    static const std::regex plaquette(R"(Plaquette: \[ [0-9]+ \] ([^ ]+))");
    static const std::regex planes(R"(Plaquette planes: ([^ ]+) ([^ ]+))");
    static const std::regex temporal(R"(Polyakov Loop: \[ [0-9]+ \] \(([^,]+),([^\)]+)\))");
    static const std::regex spatial(R"(Polyakov Loop in the ([012]) spatial direction : \[ [0-9]+ \] \(([^,]+),([^\)]+)\))");
    std::smatch match;

    if (text.find("Unsmeared plaquette") != std::string::npos) { _smearedPlaquette = false; }
    if (text.find("Smeared plaquette") != std::string::npos) { _smearedPlaquette = true; }

    if (!_smearedPlaquette && std::regex_search(text, match, plaquette))
    { _plaquette = std::stod(match[1]); }

    if (std::regex_search(text, match, planes)) {
      const double spatialPlaquette = std::stod(match[1]);
      const double temporalPlaquette = std::stod(match[2]);

      _out << "MEASplaq ss: " << spatialPlaquette
           << "  st: " << temporalPlaquette
           << "  tot: " << _plaquette << '\n';

      if (_in.ploop) { _writePolyakov(); }
    }

    if (std::regex_search(text, match, temporal)) {
      _temporalPolyakov = ComplexD(std::stod(match[1]), std::stod(match[2]));
      _spatialPolyakov = 0;
    }

    if (std::regex_search(text, match, spatial)) {
      _spatialPolyakov += ComplexD(std::stod(match[2]), std::stod(match[3]))/3.;
      if (match[1] == "2" && !_in.plaq) { _writePolyakov(); }
    }
  }

private: // Hamiltonian and acceptance records
  void _readHamiltonian(const std::string& text) {
    static const std::regex action(R"(S \[([01])\]\[[0-9]+\] H = ([^ ]+))");
    static const std::regex total(R"(Total H (before trajectory = |after trajectory  = )([^ ]+))");
    static const std::regex metropolis(R"(exp\(-dH\) = ([^ ]+)  Random = ([^ ]+))");
    std::smatch match;

    if (text.find("Compute initial action") != std::string::npos ||
        text.find("Compute final action") != std::string::npos) {
      if (_molecularDynamics) { _out << '\n'; }
      _fermion = 0;
      _gauge = 0;
      _molecularDynamics = false;
    }

    if (text.find("Molecular Dynamics evolution") != std::string::npos)
    { _molecularDynamics = true; }

    if (std::regex_search(text, match, action)) {
      const double value = std::stod(match[2]);

      if (match[1] == "0") { _gauge = value; }
      else { _fermion += value; }
    }

    if (std::regex_search(text, match, total)) {
      const double hamiltonian = std::stod(match[2]);
      const double kinetic = hamiltonian - _fermion - _gauge - 16*_volume;
      const auto deltaH = text.find("dH = ");

      _out << "kinetic: " << kinetic
           << " gauge: " << _gauge
           << " fermion: " << _fermion << " \n";

      if (match[1] == "before trajectory = ") { _out << '\n'; }
      if (deltaH != std::string::npos) { _deltaH = std::stod(text.substr(deltaH + 5)); }
    }

    if (std::regex_search(text, match, metropolis)) {
      _probability = std::stod(match[1]);
      _random = std::stod(match[2]);
    }

    if (text.find("Metropolis_test -- ") != std::string::npos ||
        text.find("Skipping Metropolis test") != std::string::npos) {
      const bool accepted = text.find("REJECTED") == std::string::npos;

      if (!_in.hmc.MetropolisTest) { _probability = std::exp(-_deltaH); _random = 0; }

      _out << (accepted ? "ACC: " : "REJ: ") << _deltaH
           << " exp(dH): " << _probability
           << " rand: " << _random << '\n';

      if (_in.info.count("measurements")) { _out << '\n'; }
      _observing = true;
    }

    if (text.find(":::::::::::::::::::::::::::::::::::::::::::") != std::string::npos &&
        _in.info.count("measurements"))
    { _out << '\n'; }
  }

private: // solver records
  bool _readSolver(const std::string& text) {
    static const std::regex summary(R"(HISQSolver: (mixed|double) iterations ([0-9]+) residual ([^ ]+) seconds ([^ ]+))");
    static const std::regex iterations(R"(ConjugateGradient Converged on iteration ([0-9]+))");
    static const std::regex residual(R"(True residual ([^ ]+))");
    static const std::regex elapsed(R"(Solver Elapsed +([0-9.eE+-]+) s)");
    std::smatch match;

    // HISQSolver supplies one summary for the complete double or mixed solve.
    if (text.find("HISQSolver: ") != std::string::npos &&
        std::regex_search(text, match, summary)) {
      const double seconds = std::stod(match[4]);

      _iterations = std::stoi(match[2]);
      _residual = std::stod(match[3]);
      _solving = false;
      _writeSolver(seconds);
      return true;
    }

    // Native Grid CG reports its results over several lines.
    if (text.find("guess is converged already") != std::string::npos) { _iterations = 0; _residual = 0; }

    if (std::regex_search(text, match, iterations)) { _iterations = std::stoi(match[1]); }
    if (std::regex_search(text, match, residual)) { _residual = std::stod(match[1]); }
    if (!_solving && std::regex_search(text, match, elapsed)) { _writeSolver(std::stod(match[1])); }

    return false;
  }

private: // gauge action and smearing timings
  bool _readGaugeTiming(const std::string& text) {
    static const std::regex timing(R"(HISQGaugeAction: (action|force) ([0-9.eE+-]+))");
    std::smatch match;

    if (text.find("HISQGaugeAction: ") == std::string::npos) { return false; }

    if (std::regex_search(text, match, timing)) {
      const auto flags = _out.flags();
      const auto precision = _out.precision();
      const double seconds = std::stod(match[2]);
      const char* label = match[1] == "action" ? "gaugeAction: " : "gaugeForce: ";

      _out << std::fixed << std::setprecision(5) << label << seconds << "s\n";

      _out.flags(flags);
      _out.precision(precision);
    }

    return true;
  }

  void _readSmearingTiming(const std::string& text) {
    static const std::regex timing(R"((Smearing in|Smeared Force chain rule took) ([0-9.eE+-]+) ms flops = ([0-9.eE+-]+))");
    std::smatch match;

    if (_observing || !std::regex_search(text, match, timing)) { return; }

    const auto flags = _out.flags();
    const auto precision = _out.precision();
    const double seconds = std::stod(match[2])*1e-6;
    const double gigaflops = seconds > 0 ? 1e-9*std::stod(match[3])/seconds : 0;
    const char* label = match[1] == "Smearing in" ? "linkSmear: " : "forceSmear: ";

    _out << std::fixed << std::setprecision(5) << label << seconds << "s "
         << std::setprecision(3) << gigaflops << "Gf/s\n";

    _out.flags(flags);
    _out.precision(precision);
  }

private: // complete log lines
  void _line(const std::string& text) {
    const bool diagnostic =
      text.find("Error") != std::string::npos ||
      text.find("failed") != std::string::npos ||
      text.find("WARNING") != std::string::npos ||
      text.find("NOT converge") != std::string::npos;

    if (text.find("Refresh momenta and pseudofermions") != std::string::npos)
    { _observing = false; }

    if (text.find("HISQSolver: begin") != std::string::npos) { _solving = true; return; }

    // Suppress inner solver logs until HISQSolver emits its summary.
    if (_solving && text.find("HISQSolver: ") == std::string::npos && !diagnostic) { return; }

    // QEX measurement and checkpoint records already have the desired format.
    for (const char* tag : {
      "MEASplaq", "MEASploop", "MEASpbp", "loading gauge file:",
      "read serial RNG file:", "read parallel RNG file:"
    }) {
      const auto start = text.find(tag);
      if (start != std::string::npos) { _out << text.substr(start) << '\n'; return; }
    }

    if (_readGaugeTiming(text)) { return; }
    _readMeasurements(text);
    _readHamiltonian(text);
    if (_readSolver(text)) { return; }
    _readSmearingTiming(text);

    if (diagnostic) { _out << text << '\n'; }
  }

private: // stream buffer
  void _finishLine() { _line(_pending); _pending.clear(); }

  std::streamsize xsputn(const char* data, std::streamsize size) override {
    const auto original = size;

    while (size > 0) {
      const char* newline = traits_type::find(data, size, '\n');
      const auto length = newline ? newline - data : size;

      _pending.append(data, length);
      data += length;
      size -= length;

      if (newline) { _finishLine(); ++data; --size; }
    }

    return original;
  }

  int overflow(int ch) override {
    if (ch != traits_type::eof()) { if (ch == '\n') { _finishLine(); } else { _pending += char(ch); } }
    return traits_type::not_eof(ch);
  }

  int sync() override { _out.flush(); return 0; }
};

void QEXOutput::parameters() {
  // ensemble and trajectory
  _out << "\nbase ensemble name: " << _in.ensemble << '\n'
       << "starting configuration: " << _in.hmc.StartTrajectory << '\n'
       << "number of trajectories: " << _in.hmc.Trajectories << "\n\n"
       << "JSON input: " << _in.info.dump() << "\n\n"
       << "trajectory length: " << _in.hmc.MD.trajL << '\n';

  // integrators
  _out << "outer (fermion/Hasenbusch) integrator: 2MN\n"
       << "outer (fermion/Hasenbusch) steps: " << _in.fermionSteps << '\n'
       << "inner (gauge) integrator: 2MN\n"
       << "inner (gauge) steps per outer (fermion/Hasenbusch) gauge update: " << _in.gaugeSteps << '\n';

  // initialization
  _out << "gauge start: " << _in.start << '\n'
       << "serial RNG: MILC\nserial seed: " << _in.serialSeed << '\n'
       << "parallel RNG: MILC\nparallel seed: " << _in.parallelSeed << '\n';

  // action and solvers
  _out << "beta: " << _in.beta << "\nmass: " << _in.mass << '\n'
       << "hasenbusch mass: " << _in.hmass << '\n'
       << "plaquette coefficient: 1.0\nrectangle coefficient: -0.05\n"
       << "unitary projection method: cayley-hamilton\n"
       << "unitary projection force cutoff: " << _in.cutoff << '\n'
       << "minimum squared residual (action CG solver): " << _in.actionR2 << '\n'
       << "maximum iterations (action CG solver): " << _in.actionMax << '\n'
       << "minimum squared residual (force CG solver): " << _in.forceR2 << '\n'
       << "maximum iterations (force CG solver): " << _in.forceMax << '\n'
       << "boundary conditions: pppa\n";

  // smearing coefficients
  for (int stage = 0; stage < 2; ++stage) {
    const double lepage = stage == 0 ? _in.lepage : 2 - _in.lepage;
    const double naik = stage == 0 ? 0 : _in.naik;

    _out << (stage == 0 ? "fat7first:" : "fat7second:") << '\n'
         << "  oneLink:     " << (1 + 3*lepage + naik)/8 << '\n'
         << "  threeStaple: " << -1./16 << '\n'
         << "  fiveStaple:  " << 1./64 << '\n'
         << "  sevenStaple: " << -1./384 << '\n'
         << "  lepage:      " << -lepage/16 << '\n';
  }

  _out << "naik: " << -_in.naik/24 << "\n\nsmearing: HISQ\n\n";
}

// Input-dependent measurement and checkpoint callbacks.
#if defined(RNG_MILCV6) && defined(HAVE_LIME)

void HISQMeasurements::_condensate(GridParallelRNG& parallel) const {
  GridBase* grid = _op.FermionGrid();
  GridBase* rbGrid = _op.FermionRedBlackGrid();
  const RealD volume = grid->gSites();
  const double factor = .5*_in.mass/volume;
  FermionField noise(grid), solution(grid), parity(rbGrid);

  // alphashmc uses fixed ActionCGTol here, independently of r2-action.
  HISQSolver CG(1e-10, 10000, nullptr, nullptr);
  SchurRedBlackStaggeredSolve<FermionField> solve(CG);

  for (int source = 0; source < _in.sources; ++source) {
    double even, odd;

    gaussian(parallel, noise);
    solution = Zero();

    if (_in.mass == 0) {
      // Schur reconstruction divides by m; use the full normal operator.
      FermionField rhs(grid);
      MdagMLinearOperator<Fermion, FermionField> normal(_op);

      _op.Mdag(noise, rhs);
      CG(normal, rhs, solution);
    } else { solve(_op, noise, solution); }

    pickCheckerboard(Even, parity, solution);
    even = norm2(parity);
    pickCheckerboard(Odd, parity, solution);
    odd = norm2(parity);

    std::cout << GridLogMessage << "MEASpbp (" << source << ") mass: " << _in.mass
              << " pbpe: " << factor*even
              << " pbpo: " << factor*odd << '\n';
  }
}

void HISQMeasurements::TrajectoryComplete(
  int traj,
  ConfigurationBase<GaugeField>& links,
  GridSerialRNG& serial,
  GridParallelRNG& parallel
) {
  if (_in.plaq) { _gauge(links.get_U(false)); }
  if (_in.sources > 0) { _op.ImportGauge(_contract); _condensate(parallel); }
}

void QEXCheckpointer::CheckpointRestore(
  int traj,
  GaugeField& U,
  GridSerialRNG& serial,
  GridParallelRNG& parallel
) {
  const auto base = _in.ensemble + "_" + std::to_string(traj);
  uint64_t cached = 0;

  readQEXGauge(U, base + ".lat");

  if (traj == 0) {
    serial.SeedFixedIntegers({_in.serialSeed, 987654321});
    parallel.SeedFixedIntegers({_in.parallelSeed});
  } else {
    qexRNGCheckpoint(base, serial, parallel, true);

    // alphashmc checkpoints after whole-field Gaussian draws (even counts).
    // Grid resets its cache before each fill, so a mid-draw checkpoint would
    // discard an unconsumed QEX Gaussian.
    cached = serial._gaussian[0]._hasSpare;
    for (const auto& distribution : parallel._gaussian) { cached += distribution._hasSpare; }

    U.Grid()->GlobalSum(cached);
    require(cached == 0, "Cannot resume a mid-draw QEX Gaussian cache with Grid's fill interface");
  }
}

void QEXCheckpointer::TrajectoryComplete(
  int traj,
  ConfigurationBase<GaugeField>& links,
  GridSerialRNG& serial,
  GridParallelRNG& parallel
) {
  const auto base = _in.ensemble + "_" + std::to_string(traj);
  if (_in.save <= 0 || traj%_in.save != 0) { return; }
  writeQEXGauge(links.get_U(false), base + ".lat");
  qexRNGCheckpoint(base, serial, parallel, false);
}

#endif

//
// main program
//

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  QEXInput input(argc, argv);
  HMCWrapper TheHMC(input.hmc);
  GridModule Grid;
  CheckpointerParameters Checkpoint;
  RNGModuleParameters Rng;

  // Grid setup
  Grid.set_full(SpaceTimeGrid::makeFourDimGrid(input.lattice,input.simd,input.ranks));
  Grid.set_rb(SpaceTimeGrid::makeFourDimRedBlackGrid(Grid.get_full()));
  TheHMC.Resources.AddGrid("gauge", Grid);
  auto GridPtr = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();

  // quick checks
  if (GridPtr->IsBoss()) {
    if (input.reverse > 0)
    { std::cerr << "HISQ4f: reversibility-check is unsupported; requested checks will be skipped.\n"; }
    if (input.hmc.StartingType == "HotStart")
    { std::cerr << "HISQ4f: fresh hot starts use Grid's initialization and differ from QEX's.\n"; }
  }

  // Mixed precision grids
  std::unique_ptr<GridCartesian> singleGrid;
  std::unique_ptr<GridRedBlackCartesian> singleRB;
  if (input.mixedPrecision) { 
    singleGrid.reset(SpaceTimeGrid::makeFourDimGrid(input.lattice, input.simdSingle,input.ranks));
    singleRB.reset(SpaceTimeGrid::makeFourDimRedBlackGrid(singleGrid.get()));
  }

  // Checkpointer setup
  Checkpoint.config_prefix = input.ensemble+".lat";
  Checkpoint.rng_prefix = input.ensemble+".rng";
  Checkpoint.saveInterval = input.save > 0 ? input.save : std::numeric_limits<int>::max();
  Checkpoint.saveSmeared = false;
  TheHMC.Resources.LoadScidacCheckpointer(Checkpoint, QEXCheckpointFormat{input});
  
  // RNG seeds
  Rng.serial_seeds = std::to_string(input.serialSeed) + " 987654321";
  Rng.parallel_seeds = std::to_string(input.parallelSeed);
  TheHMC.Resources.SetRNGSeeds(Rng);
  
  // HISQ smearing
  HISQConfiguration<Gimpl> HISQSmearing(GridPtr, input.context());
  HISQSmearing.setPerformanceReporting(true);
  
  // full action parameters
  const double cp = 1.0;
  const double cr = -1./20.;
  const double cpg = 0.0;
  const double c1 = 1.0;
  const double c2 = -input.naik/24.0;
  const double u0 = 1.0;

  // gauge action
  HISQGaugeAction GaugeAction(GridPtr, input.beta, cp, cr, cpg);

  // fermion operators
  Fermion::ImplParams params(std::vector<Complex>{1,1,1,-1});
  Fermion light(*GridPtr, *GridRBPtr, input.mass, c1, c2, u0,params);
  Fermion heavy(*GridPtr, *GridRBPtr, input.hmass, c1, c2, u0,params); // hasenbusch
  
  // mixed precision setup
  std::unique_ptr<FermionF> lightF, heavyF;
  if (input.mixedPrecision) { 
    lightF = std::make_unique<FermionF>(*singleGrid, *singleRB, input.mass, c1, c2, u0, params);
    heavyF = std::make_unique<FermionF>(*singleGrid, *singleRB, input.hmass, c1, c2, u0, params);
  }

  // solver setup
  HISQSolver lightActionCG(std::sqrt(input.actionR2), input.actionMax, lightF.get(), &light);
  HISQSolver lightForceCG(std::sqrt(input.forceR2), input.forceMax, lightF.get(), &light);
  HISQSolver heavyActionCG(std::sqrt(input.actionR2), input.actionMax, heavyF.get(), &heavy);
  HISQSolver heavyForceCG(std::sqrt(input.forceR2), input.forceMax, heavyF.get(), &heavy);
  HISQSolver heatbathCG(std::sqrt(input.actionR2), input.actionMax, nullptr, nullptr);
  
  // pseudofermion actions
  PseudoFermionActionReg regulator(heavy, heavyForceCG, heavyActionCG);
  PseudoFermionActionRat ratio(heavy, light, lightForceCG, lightActionCG, heatbathCG);
  
  // smearing contracts
  ratio.signContract(HISQSmearing.promise(), HISQSmearing.promise()); 
  regulator.signContract(HISQSmearing.promise());

  // composite pseudofermion class
  HISQPseudoFermions Fermions;
  Fermions.push_back(&regulator); 
  Fermions.push_back(&ratio);

  // integrator setup
  ActionLevel<GaugeField> Level1(input.gaugeSteps), Level2(input.fermionSteps);
  auto integrator = [&](auto& md) {
    md.schemes[0] = omelyan2(md.T, md.V[0], input.gaugeSteps);
    md.schemes[1] = omelyan2(md.schemes[0], md.V[1], input.fermionSteps);
    md.evolution.set_branches({&md.schemes[1]});
  };
  Level1.push_back(&GaugeAction); 
  Level2.push_back(&Fermions);
  TheHMC.TheAction.push_back(Level1); 
  TheHMC.TheAction.push_back(Level2);

  // measurement setup
  if (input.plaq) { TheHMC.Resources.AddObservable<PlaquetteMod<Gimpl>>(); }
  if (input.ploop) {
    TheHMC.Resources.AddObservable<PolyakovMod<Gimpl>>();
    TheHMC.Resources.AddObservable<SpatialPolyakovMod<Gimpl>>();
  }
  if (input.plaq || input.sources>0) 
  { TheHMC.Resources.AddObservable<HISQMeasurementModule>(input, light, HISQSmearing); }
  
  // finally run the HMC
  QEXOutput output(input, GridPtr);
  TheHMC.Run(HISQSmearing, integrator);

  Grid_finalize();
}
