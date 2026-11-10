/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./HMC/HISQ2p1p1f.cc

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

/**
 * @file HISQ2p1p1f.cc
 * @author Curtis Taylor Peterson
 * @brief RHMC for 2+1+1 HISQ fermions
 * @details The action is specified in Appendix A of
 * "Scaling studies of QCD with the dynamical highly improved staggered quark action",
 * https://doi.org/10.1103/PhysRevD.82.074501.
 * The five-field Hasenbusch factorization follows Sec. II of that paper
 * and the physical-mass ensembles in https://doi.org/10.1103/PhysRevD.87.054505:
 * one combined light/strange ratio, three one-flavor regulators, and charm.
 * The pseudofermions follow MILC's ks_imp_rhmc rationals file (e.g.
 * ks_imp_rhmc/test/rationals.sample.su3_rhmc_hisq_f211): the combined field is one
 * rational function of -D^2, with the light, strange and regulator masses entering as
 * constant shifts, and every field shares MILC's bounds on -D^2 with its own MD (force)
 * and GR/FA (heatbath, action) orders.
 */

#include <Grid/Grid.h>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <chrono>
#include <iomanip>
#include <streambuf>
#include <deque>
#include <limits>
#include <map>
#include <regex>
#include <sstream>

using namespace Grid;

////////////////////////////////////////////////////////////////
// Driver types and MILC records
////////////////////////////////////////////////////////////////

// MILC's node0_printf format, kept verbatim by MILCOutput
template <class... Args>
std::string milcRecord(const char* format, Args... args) {
  std::string record(std::snprintf(nullptr, 0, format, args...), '\0');
  std::snprintf(record.data(), record.size() + 1, format, args...);
  return record;
}

class MILCGaugeImpl: public PeriodicGimplR {
public:
  inline static bool report = false;

  static void update_field(Field& P, Field& U, double ep) {
    const double start = report ? usecond() : 0.0;
    PeriodicGimplR::update_field(P, U, ep);
    if (report) std::cout << GridLogMessage << milcRecord(
      "LINK_UPDATE: time = %e (Grid update_field D)", (usecond() - start)*1.0e-6
    ) << std::endl;
  }
};

typedef HMCWrapperTemplate<MILCGaugeImpl, InterleavedIntegrator> HMCWrapper;
typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;

typedef ImprovedStaggeredFermionD StaggeredFermionOperator;
typedef StaggeredImplD FermionImplPolicy;
typedef StaggeredEvenEvenRational<FermionImplPolicy> FermionAction;

typedef PeriodicGimplD Gimpl;

typedef XmlReader Serialiser;

////////////////////////////////////////////////////////////////
// MILC trajectory output, scoped to this driver's Run call
////////////////////////////////////////////////////////////////

// MILC's performance records depend on its solver/backend and compile-time
// CGTIME/FLTIME/FFTIME options. Use Grid's staggered solver work models;
// mixed output also retains the original residuals and counters.
class MILCPerformanceFormatter {
  std::ostream& out_;
  int steps_, warmupsEnd_, interval_, trajectory_ = -1;
  double localVolume_;
  bool md_ = false, warmupsPrinted_ = false;
  long long totalIterations_ = 0, mdIterations_ = 0;
  double measuredIterationsPerStep_ = 0.0;
  int measurements_ = 0;
  // Solves are synchronous; keep the current solve until its elapsed-time record.
  struct Solve {
    int masses = 0, iterations = 0;
    bool converged = false, reported = false;
  } solve_;
  std::string timerSection_;
  double gaugeActionFlops_ = 0.0, gaugeForceFlops_ = 0.0;
  double pendingGaugeSeconds_ = -1.0;
  int gaugeActionCalls_ = 0;
  bool gaugeSize_ = false;
  int refreshing_ = -1;
  std::map<int, double> refreshSolverSeconds_;

  void work(double flops, double seconds) {
    if (flops <= 0.0) return;
    out_ << " flops/site = " << flops/localVolume_ << " flops = " << flops;
    if (seconds > 0.0) out_ << " mflops = " << flops/(1.0e6*seconds);
  }

  void cg(double seconds, const char* kind, bool multishift = false) {
    if (!solve_.converged || solve_.reported) return;
    // Test_staggered_multishift.cc: (1205+15*degree)*V*iterations.
    // Test_staggered_cg_schur.cc: 1146*V*iterations (DeoDoe only).
    // MILC reports throughput per rank; these are analytic work models, not
    // hardware instruction counts. Solver elapsed excludes residual checking.
    const double perSite = multishift ? 1205.0 + 15.0 * solve_.masses : 1146.0;
    const double flops = perSite * localVolume_ * solve_.iterations;
    out_ << std::scientific << std::setprecision(6) << "CONGRAD5:";
    if (seconds >= 0.0) out_ << " time = " << seconds;
    out_ << " (Grid " << kind << " D) masses = " << solve_.masses
         << " iters = " << solve_.iterations;
    if (seconds > 0.0) out_ << " mflops = " << flops / (1.0e6 * seconds);
    out_ << " flops = " << flops << '\n';
    if (multishift && refreshing_ >= 0) refreshSolverSeconds_[refreshing_] += seconds;
    solve_.reported = true;
  }
  void iterations(int n) {
    if (solve_.converged) return;
    solve_.iterations = n;
    solve_.converged = true;
    totalIterations_ += n;
    if (md_) mdIterations_ += n;
  }

public:
  MILCPerformanceFormatter(std::ostream& out, int steps, int start, int warmups, int interval,
                           double volume = 1.0, int ranks = 1)
    : out_(out), steps_(steps), warmupsEnd_(start + warmups), interval_(interval),
      localVolume_(volume / ranks) {}

  void line(const std::string& text) {
    std::smatch m;
    static const std::regex trajectory("-- # Trajectory = ([0-9]+)");
    if (std::regex_search(text, m, trajectory)) {
      trajectory_ = std::stoi(m[1]);
      mdIterations_ = 0;
      md_ = false; refreshing_ = -1; refreshSolverSeconds_.clear();
      timerSection_.clear();
      gaugeActionCalls_ = 0; pendingGaugeSeconds_ = -1.0;
      gaugeSize_ = false;
      if (trajectory_ >= warmupsEnd_) { finishWarmups(); }
    }
    static const std::regex refresh("refresh \\[([0-9]+)\\]\\[([0-9]+)\\]");
    if (std::regex_search(text, m, refresh)) refreshing_ = m[1] == "0" ? std::stoi(m[2]) : -1;
    if (text.find("Compute initial action") != std::string::npos) refreshing_ = -1;
    if (text.find("Molecular Dynamics evolution") != std::string::npos) md_ = true;
    if (text.find("Compute final action") != std::string::npos) md_ = false;

    static const std::regex shift("ConjugateGradientMultiShift: shift ([0-9]+) target resid");
    if (std::regex_search(text, m, shift)) {
      const int index = std::stoi(m[1]);
      if (index == 0) solve_ = Solve{};
      solve_.masses = std::max(solve_.masses, index + 1);
    }
    static const std::regex multiDone("CGMultiShift: All shifts have converged iteration ([0-9]+)");
    if (std::regex_search(text, m, multiDone)) iterations(std::stoi(m[1]));
    if (text.find("ConjugageGradient::LogBegin") != std::string::npos ||
        text.find("ConjugateGradient::LogBegin") != std::string::npos) solve_ = Solve{};
    static const std::regex singleDone("ConjugateGradient Converged on iteration ([0-9]+)");
    if (std::regex_search(text, m, singleDone)) {
      solve_ = Solve{};
      solve_.masses = 1;
      iterations(std::stoi(m[1]));
    }
    if (text.find("ConjugateGradient guess is converged already") != std::string::npos) {
      solve_ = Solve{};
      solve_.masses = 1;
      iterations(0);
      cg(-1.0, "cg already-converged");
    }
    static const std::regex solverTime("(Solver Elapsed|[ :\\t]Elapsed) +([0-9.eE+-]+) s");
    if (std::regex_search(text, m, solverTime)) {
      const bool multishift = m[1] != "Solver Elapsed";
      cg(std::stod(m[2]), multishift ? "multicg" : "cg", multishift);
    }

    static const std::regex model("GaugeWorkModel: action = ([0-9.eE+-]+) force = ([0-9.eE+-]+)");
    if (text.find("GaugeWorkModel:") != std::string::npos && std::regex_search(text, m, model)) {
      gaugeActionFlops_ = std::stod(m[1]); gaugeForceFlops_ = std::stod(m[2]);
    }
    if (text.find("S [1][0] H = ") != std::string::npos) ++gaugeActionCalls_;
    // HISQ's existing logs label usecond() differences as "ms"; convert microseconds.
    static const std::regex hisqTime(
      "(Smearing in|Smeared Force chain rule took) ([0-9.eE+-]+) ms"
      "( flops = ([0-9.eE+-]+) sites = ([0-9.eE+-]+))?");
    if ((text.find("Smearing in ") != std::string::npos ||
         text.find("Smeared Force chain rule took ") != std::string::npos) &&
        std::regex_search(text, m, hisqTime)) {
      const bool smear = m[1] == "Smearing in";
      const double seconds = std::stod(m[2])*1.0e-6;
      out_ << std::scientific << std::setprecision(6)
           << (smear ? "FLTIME" : "FFTIME") << ": time = " << seconds << " (HISQ Grid";
      if (m[4].matched) { out_ << " analytic-core"; }
      if (!smear) { out_ << " pullback"; }
      out_ << " D)";
      if (m[4].matched) { work(std::stod(m[4]), seconds); }
      out_ << '\n';
    }
    static const std::regex trajectoryTime("Total time for trajectory \\(s\\): ([0-9.eE+-]+)");
    if (std::regex_search(text, m, trajectoryTime)) {
      out_ << std::scientific << std::setprecision(6)
           << "Aggregate time to do one trajectory " << std::stod(m[1]) << '\n';
      const int measuring = trajectory_ + 1 - warmupsEnd_;
      if (steps_ > 0 && interval_ > 0 && measuring > 0 && measuring % interval_ == 0) {
        // MILC update() returns integer iters/steps before control.c averages.
        measuredIterationsPerStep_ += mdIterations_ / steps_;
        ++measurements_;
      }
    }
    if (text.find("Refresh cumulative timings") != std::string::npos) timerSection_ = "refresh";
    if (text.find("Action cumulative timings") != std::string::npos) timerSection_ = "action";
    if (text.find("Force cumulative timings") != std::string::npos) timerSection_ = "force";
    if (text.find("Dslash counts") != std::string::npos) timerSection_.clear();
    static const std::regex actionTimer("([A-Za-z][A-Za-z0-9_]*)\\[([0-9]+)\\]\\[([0-9]+)\\] +([0-9.eE+-]+) s");
    if (!timerSection_.empty() && std::regex_search(text, m, actionTimer)) {
      const bool gauge = m[2] == "1";
      const char* tag = timerSection_ == "force" ? (gauge ? "GFTIME" : "FFTIME") : "GRID_ACTIONTIME";
      const double seconds = std::stod(m[4]);
      if (gauge && timerSection_ == "force") pendingGaugeSeconds_ = seconds;
      else {
        out_ << std::scientific << std::setprecision(6)
             << tag << ": time = " << seconds << " (Grid cumulative "
             << timerSection_ << " D) level = " << m[2] << " field = " << m[3];
        if (gauge && timerSection_ == "action") work(gaugeActionFlops_*gaugeActionCalls_, seconds);
        out_ << '\n';
        if (!gauge && timerSection_ == "refresh") {
          const double cg = refreshSolverSeconds_[std::stoi(m[3])];
          out_ << "GRSOURCETIME: time = " << std::max(0.0, seconds-cg)
               << " (Grid refresh excluding timed CG loop D) field = " << m[3]
               << " cg_time = " << cg << " refresh_time = " << seconds << '\n';
        }
      }
    }
    if (text.find("][0] :") != std::string::npos)
      gaugeSize_ = text.find("GaugeAction[1][0] :") != std::string::npos;
    static const std::regex calls("calls +([0-9]+)");
    if (gaugeSize_ && pendingGaugeSeconds_ >= 0.0 && std::regex_search(text,m,calls)) {
      out_ << std::scientific << std::setprecision(6) << "GFTIME: time = " << pendingGaugeSeconds_
           << " (Grid cumulative force D) level = 1 field = 0";
      work(gaugeForceFlops_*std::stoll(m[1]),pendingGaugeSeconds_);
      out_ << '\n'; pendingGaugeSeconds_ = -1.0;
    }
  }

  void finishWarmups() {
    if (!warmupsPrinted_) {
      out_ << "WARMUPS COMPLETED\n";
      warmupsPrinted_ = true;
    }
  }
  void complete() {
    if (pendingGaugeSeconds_ >= 0.0) {
      out_ << std::scientific << std::setprecision(6) << "GFTIME: time = " << pendingGaugeSeconds_
           << " (Grid cumulative force D) level = 1 field = 0\n";
      pendingGaugeSeconds_ = -1.0;
    }
    if (measurements_ > 0) {
      out_ << std::scientific << std::setprecision(6)
           << "average cg iters for step= " << measuredIterationsPerStep_ / measurements_ << '\n';
    }
  }
  long long totalIterations() const { return totalIterations_; }
};

// Interpret Grid trajectory records independently of stream ownership.
class MILCTrajectoryFormatter {
  std::ostream& output_;
  MILCPerformanceFormatter performance_;
  const bool pure_;
  const size_t fermions_;
  const double kineticOffset_;
  std::map<int, double> fermionTerms_;
  double gauge_ = 0.0, deltaH_ = 0.0;
  double previousGauge_ = 0.0, previousKinetic_ = 0.0, previousFermion_ = 0.0;
  double previousTotal_ = 0.0;
  bool haveGauge_ = false, evaluating_ = false, haveDeltaH_ = false;
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_ = Clock::now(), actionStart_;

  void beginAction() {
    fermionTerms_.clear();
    haveGauge_ = false;
    evaluating_ = true;
    actionStart_ = Clock::now();
  }

  void action(double total) {
    if (!evaluating_ || !haveGauge_ || fermionTerms_.size() != fermions_) {
      output_ << "WARNING: incomplete Grid action records; ACTION summary omitted\n";
      evaluating_ = false;
      return;
    }
    double fermion = 0.0;
    for (size_t i = 0; i < fermions_; ++i) {
      auto term = fermionTerms_.find(i);
      if (term == fermionTerms_.end()) {
        output_ << "WARNING: missing pseudofermion action; ACTION summary omitted\n";
        evaluating_ = false;
        return;
      }
      fermion += term->second;
    }
    const double kinetic = total - gauge_ - fermion - kineticOffset_;
    const double sum = total - kineticOffset_;
    output_ << std::scientific << std::setprecision(14)
            << "ACTION: g,h,f = " << gauge_ << "  " << kinetic << "  "
            << fermion << "  " << sum << '\n';
    output_ << std::setprecision(6)
            << "DG = " << gauge_ - previousGauge_
            << ", DH = " << kinetic - previousKinetic_
            << ", DF = " << fermion - previousFermion_
            << ", D = " << sum - previousTotal_ << '\n'
            << "ACTIONTIME: time = "
            << std::chrono::duration<double>(Clock::now() - actionStart_).count() << '\n';
    previousGauge_ = gauge_;
    previousKinetic_ = kinetic;
    previousFermion_ = fermion;
    previousTotal_ = sum;
    evaluating_ = false;
  }

  void decision(const char* label) {
    if (haveDeltaH_) {
      output_ << std::scientific << std::setprecision(6)
              << label << ": delta S = " << deltaH_ << '\n';
    } else {
      output_ << "WARNING: missing Grid dH; acceptance summary omitted\n";
    }
    haveDeltaH_ = false;
    output_.flush();
  }

public:
  MILCTrajectoryFormatter(std::ostream& output, size_t fermions, double volume,
                          int steps = 1, int start = 0, int warmups = 0, int interval = 1, int ranks = 1,
                          bool pure = true)
    : output_(output), performance_(output, steps, start, warmups, interval, volume, ranks),
      pure_(pure), fermions_(fermions), kineticOffset_(16.0 * volume) {}

  // Pure output keeps real diagnostics while suppressing the original detail
  // records. Reuse this path for an unterminated diagnostic during unwinding.
  void detail(const std::string& text) {
    if (!pure_) output_ << text << '\n';
    else {
      static const std::regex diagnostic(
        "\\b(warning|error|fatal|failed|failure|nan|inf)\\b|not.*converg",
        std::regex::icase);
      if (std::regex_search(text, diagnostic)) output_ << text << '\n';
    }
  }

  void line(const std::string& text) {
    if (!pure_) output_ << text << '\n';
    performance_.line(text);
    if (text.find("Compute initial action") != std::string::npos ||
        text.find("Compute final action") != std::string::npos) beginAction();
    if (text.find("-- # Trajectory = ") != std::string::npos) haveDeltaH_ = false;

    static const std::regex term("S \\[([0-9]+)\\]\\[([0-9]+)\\] H = ([^ ]+)");
    std::smatch match;
    if (evaluating_ && std::regex_search(text, match, term)) {
      std::istringstream value(match[3].str());
      double number;
      if (value >> number) {
        if (match[1] == "0") fermionTerms_[std::stoi(match[2])] = number;
        if (match[1] == "1" && match[2] == "0") { gauge_ = number; haveGauge_ = true; }
      }
    }
    const auto totalPos = text.find("Total H ");
    if (totalPos != std::string::npos) {
      const auto equals = text.find('=', totalPos);
      if (equals != std::string::npos) {
        double total;
        std::istringstream value(text.substr(equals + 1));
        if (value >> total) action(total);
      }
      const auto deltaPos = text.find("dH = ", totalPos);
      if (deltaPos != std::string::npos) {
        std::istringstream value(text.substr(deltaPos + 5));
        haveDeltaH_ = bool(value >> deltaH_);
      }
    }
    if (text.find("Metropolis_test -- ACCEPTED") != std::string::npos) decision("ACCEPT");
    if (text.find("Metropolis_test -- REJECTED") != std::string::npos) decision("REJECT");
    if (text.find("Skipping Metropolis test") != std::string::npos) decision("CHECK");

    // Mixed mode forwards original Grid lines unchanged. Also emit a plain
    // MILC measurement record when its original carries a Grid prefix.
    static const std::regex record(
      "(PLAQ:|P_LOOP:|G_LOOP:|GACTION:|GMES:|PBP:|FACTION:|PLAQUETTE ACTION:|LINK_UPDATE:|"
      "hisq_svd_counter = |hisq_force_filter_counter = |"
      "Aggregate time to (setup|do gauge measurement|do pbp measurements) )");
    if (std::regex_search(text, match, record)) {
      if (pure_ || match.position() != 0) output_ << text.substr(match.position()) << '\n';
    } else if (pure_) detail(text);
  }

  void complete() {
    performance_.finishWarmups();
    output_ << "RUNNING COMPLETED\n";
    performance_.complete();
    output_ << std::scientific << std::setprecision(6)
            << "Time = " << std::chrono::duration<double>(Clock::now() - start_).count()
            << " seconds\n"
            << "total_iters = " << performance_.totalIterations() << '\n';
    output_.flush();
  }
};

// Collect whole lines: Grid emits a record through many operator<< calls and
// sometimes flushes before its newline. Never interpret a flush as a record.
class MILCOutput : public std::streambuf {
  std::streambuf* original_ = nullptr;
  std::ostream output_;
  std::ios::fmtflags flags_;
  std::streamsize precision_;
  std::ios::iostate state_;
  int messageActive_, hmcActive_, performanceActive_, integratorActive_;
  MILCTrajectoryFormatter formatter_;
  std::string pending_;

  void finishLine() {
    if (pending_.find("Compute initial action") != std::string::npos ||
        pending_.find("Compute final action") != std::string::npos)
      std::cout.precision(std::numeric_limits<double>::max_digits10);
    // The formatter writes to the original stream, so the buffer can be reused.
    formatter_.line(pending_);
    pending_.clear();
  }

protected:
  std::streamsize xsputn(const char* data, std::streamsize size) override {
    // std::ostream sends strings/numbers in blocks. Avoid a virtual overflow
    // call for every byte; retain partial records across writes and flushes.
    const std::streamsize originalSize = size;
    while (size > 0) {
      const char* newline = traits_type::find(data, size, '\n');
      const std::streamsize length = newline ? newline-data : size;
      pending_.append(data, length);
      data += length;
      size -= length;
      if (newline) { finishLine(); ++data; --size; }
    }
    return originalSize;
  }
  int_type overflow(int_type ch) override {
    if (!traits_type::eq_int_type(ch, traits_type::eof())) {
      const char c = traits_type::to_char_type(ch);
      if (c == '\n') finishLine();
      else pending_ += c;
    }
    return traits_type::not_eof(ch);
  }
  int sync() override { return output_.rdbuf()->pubsync(); }

public:
  MILCOutput(bool enabled, size_t fermions, double volume,
             int steps = 1, int start = 0, int warmups = 0, int interval = 1, int ranks = 1,
             bool pure = true)
    : output_(std::cout.rdbuf()), flags_(std::cout.flags()),
      precision_(std::cout.precision()), state_(std::cout.rdstate()),
      messageActive_(GridLogMessage.isActive()), hmcActive_(GridLogHMC.isActive()),
      performanceActive_(GridLogPerformance.isActive()), integratorActive_(GridLogIntegrator.isActive()),
      formatter_(output_, fermions, volume, steps, start, warmups, interval, ranks, pure) {
    if (enabled) {
      // Capture needs these channels even when --log requests NoMessage/NoHMC.
      GridLogMessage.Active(1);
      GridLogHMC.Active(1);
      GridLogPerformance.Active(1);
      GridLogIntegrator.Active(1);
      std::cout.flush();
      original_ = std::cout.rdbuf(this);
      std::cout.precision(std::numeric_limits<double>::max_digits10);
    }
  }
  MILCOutput(const MILCOutput&) = delete;
  MILCOutput& operator=(const MILCOutput&) = delete;
  void complete() {
    if (original_) {
      if (!pending_.empty()) { finishLine(); }
      formatter_.complete();
    }
  }
  ~MILCOutput() override {
    if (original_) {
      // Preserve a final diagnostic fragment during exception unwinding, but
      // never claim successful completion from a destructor.
      if (!pending_.empty()) formatter_.detail(pending_);
      output_.flush();
      std::cout.rdbuf(original_);
      std::cout.flags(flags_);
      std::cout.precision(precision_);
      std::cout.clear(state_);
      GridLogMessage.Active(messageActive_);
      GridLogHMC.Active(hmcActive_);
      GridLogPerformance.Active(performanceActive_);
      GridLogIntegrator.Active(integratorActive_);
    }
  }
};

////////////////////////////////////////////////////////////////
// MILC's combined light/strange/regulator pseudofermion
////////////////////////////////////////////////////////////////

// det(Q_f)^(Flavors_f/4) by bare mass
struct DeterminantFactor { RealD Mass; int Flavors; };

MultiShiftFunction combinedRationalFunction(
  const std::vector<DeterminantFactor>& factors,
  const StaggeredRationalActionParams& params,
  int invPow,
  bool action,
  bool inverse
) {
  const int degree = action ? params.action_degree : params.md_degree;
  const RealD tolerance = action ? params.action_tolerance : params.md_tolerance;

  std::vector<MultiShiftFunction> parts;
  RealD norm = 1.0;
  
  for (auto& factor : factors) {
    const RealD shift = 4.0 * factor.Mass * factor.Mass;
    const int power = inverse ? -factor.Flavors : factor.Flavors;
    
    std::cout << GridLogMessage
              << "Generating degree " << degree << " approximation for (x+" << shift
              << ")^(" << power << "/" << invPow << ")" << std::endl;
    
    AlgRemez remez(params.lo + shift, params.hi + shift, params.precision);
    
    double error = remez.generateApprox(degree, std::abs(factor.Flavors), invPow);
    if (error > tolerance) {
      std::cout << GridLogMessage
                << "WARNING: Remez approximation has a larger error "
                << error << " than the CG tolerance " << tolerance
                << "! Try increasing the number of poles" << std::endl;
    }
    
    parts.emplace_back(remez, tolerance, power < 0);
    norm *= parts.back().norm;
    for (auto& pole : parts.back().poles) { pole += shift; }
  }

  std::vector<std::pair<RealD, RealD>> terms; // (pole, residue)
  for (int f = 0; f < parts.size(); ++f) {
    for (int k = 0; k < parts[f].poles.size(); ++k) {
      const RealD pole = parts[f].poles[k];
      RealD residue = parts[f].residues[k];
      for (int g = 0; g < parts.size(); ++g) 
      { if (g != f) { residue *= parts[g].approx(-pole); } }
      terms.push_back({pole, residue});
  } }
  std::sort(terms.begin(), terms.end()); // smallest pole first for ConjugateGradientMultiShift

  MultiShiftFunction combined(terms.size(), params.lo, params.hi);
  combined.order = terms.size();
  combined.norm = norm;
  for (int i = 0; i < terms.size(); ++i) {
    combined.poles[i] = terms[i].first;
    combined.residues[i] = terms[i].second;
    combined.tolerances[i] = tolerance;
  }
  return combined;
}

////////////////////////////////////////////////////////////////
// Pseudofermion fields sharing one HISQ pullback per force
////////////////////////////////////////////////////////////////

// The HISQ configuration as seen by the fields' forces: pullbacks are recorded, summed
// and returned as zero forces; everything else is the HISQ configuration's own
class DeferredPullback : public ConfigurationBase<HMCWrapper::Field> {
  typedef HMCWrapper::Field Field;

  ConfigurationBase<Field>& _links;
  mutable PrimalCotangentPairs<Field> _pending;

public:
  explicit DeferredPullback(ConfigurationBase<Field>& links): _links(links) {}

  PrimalCotangentPairs<Field>& pending() { return _pending; }

  void pullback(Field& UdSdU, PrimalCotangentPairs<Field>& dSdU) const override
  { _pending += std::move(dSdU); UdSdU = Zero(); }

  void set_Field(Field& U) override { _links.set_Field(U); }
  void smeared_force(Field& UdSdU) override { _links.smeared_force(UdSdU); }
  Field& get_SmearedU() override { return _links.get_SmearedU(); }
  Field& get_U(bool smeared = false) override { return _links.get_U(smeared); }
  void smear() override { _links.smear(); }
  Primal<Field> primal(std::size_t id) const override { return _links.primal(id); }
  const Field& fundamental() const override { return _links.fundamental(); }
  const Field resolve(std::size_t id) const override { return _links.resolve(id); }
};

// MILC applies the HISQ chain rule once per force to the sum over all pseudofermions.
// Grouping the fields into one action does the same: each field's force stops at its
// cotangents (on the DeferredPullback), and their sum is pulled back once. Refresh and
// action evaluations are those of the individual fields.
class HISQPseudoFermions : public Action<HMCWrapper::Field> {
  typedef HMCWrapper::Field Field;

  std::vector<Action<Field>*> _fields;

public:
  using Action<Field>::Sinitial;

  void push_back(Action<Field>* field) {
    GRID_ASSERT(field->hasOptedIn() && "fields must use the HISQ link interface (signContract)");
    _fields.push_back(field);
  }

  std::string action_name() override { return "HISQPseudoFermions"; }

  std::string LogParameters() override {
    std::string parameters;
    for (auto* field : _fields) parameters += field->LogParameters();
    return parameters;
  }

  // through the HISQ configuration only
  void refresh(const Field&, GridSerialRNG&, GridParallelRNG&) override
  { GRID_ASSERT(0 && "HISQPseudoFermions needs the HISQ configuration"); }
  RealD S(const Field&) override
  { GRID_ASSERT(0 && "HISQPseudoFermions needs the HISQ configuration"); return 0.0; }
  void deriv(const Field&, Field&) override
  { GRID_ASSERT(0 && "HISQPseudoFermions needs the HISQ configuration"); }

  void refresh(ConfigurationBase<Field>& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG) override
  { for (auto* field : _fields) field->refresh(U, sRNG, pRNG); }

  RealD S(ConfigurationBase<Field>& U) override { return sum(U, false); }
  RealD Sinitial(ConfigurationBase<Field>& U) override { return sum(U, true); }

  void deriv(ConfigurationBase<Field>& U, Field& UdSdU) override {
    DeferredPullback deferred(U);
    Field zero(UdSdU.Grid());
    for (auto* field : _fields) field->deriv(deferred, zero); // cotangents only
    U.pullback(UdSdU, deferred.pending());                     // one HISQ chain rule
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
};

////////////////////////////////////////////////////////////////
// Serializable structs (for reading inputs)
////////////////////////////////////////////////////////////////

#define GRID_SERIALIZABLE(StructName, ...)                                        \
  struct StructName: Serializable {                                               \
    GRID_SERIALIZABLE_CLASS_MEMBERS(StructName, __VA_ARGS__);                     \
    template <class ReaderClass>                                                  \
    StructName(Reader<ReaderClass>& Reader) { read(Reader, #StructName, *this); } \
  };

GRID_SERIALIZABLE(InitializationParameters, std::string, StartType);

GRID_SERIALIZABLE(
  CheckpointParameters,
  std::string, Format,
  std::string, ConfigurationPrefix,
  std::string, RandomNumberGeneratorPrefix,
  int, SaveInterval
);

GRID_SERIALIZABLE(
  RandomNumberGeneratorParameters,
  std::string, SerialSeeds,
  std::string, ParallelSeeds
);

GRID_SERIALIZABLE(
  HamiltonianMonteCarloParameters,
  int, NoMetropolisUntil,
  
  double, TrajectoryLength,
  int, MolecularDynamicsSteps,
  
  int, FermionActionStepMultiplicity,
  int, GaugeActionStepMultiplicity
);

GRID_SERIALIZABLE(
  ActionParameters,
  double, BareGaugeCoupling,
  double, TadpoleFactor,
  double, LightFermionMass,
  double, StrangeFermionMass,
  double, CharmFermionMass,
  double, RegulatorFermionMass
);

GRID_SERIALIZABLE(
  ConjugateGradientParameters,
  double, ActionStoppingCondition,
  double, ForceStoppingCondition,
  int, MaxIterations
);

// MILC rationals file: lambda_low/lambda_high bound -D^2 for every pseudofermion;
// each pseudofermion has its own order_MD (force) and order_GR = order_FA (heatbath, action)
GRID_SERIALIZABLE(
  RationalApproximationParameters,
  double, LambdaLow,
  double, LambdaHigh,
  int, Precision,
  int, LightStrangeActionDegree,
  int, LightStrangeForceDegree,
  int, RegulatorActionDegree,
  int, RegulatorForceDegree,
  int, CharmActionDegree,
  int, CharmForceDegree
);

#undef GRID_SERIALIZABLE

// Optional: without it the gauge observables are measured every trajectory
struct ChiralCondensateParameters: Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(
    ChiralCondensateParameters,
    double, Mass,
    double, NaikEpsilon,
    double, StoppingCondition,
    int, MaxIterations
  );
};

struct MeasurementParameters: Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(
    MeasurementParameters,
    int, TrajectoriesBetweenMeasurements,
    int, ChiralCondensateSources,
    std::vector<ChiralCondensateParameters>, ChiralCondensate
  );
};

////////////////////////////////////////////////////////////////
// Input deck: the XML file, or MILC's input deck and rationals file
////////////////////////////////////////////////////////////////

// One pseudofermion: the Naik epsilon of its operator and its three rational functions of
// -D^2, generated from the Remez settings in Params (XML) or read from MILC's rationals file
struct FieldSpec {
  RealD NaikEpsilon;
  StaggeredRationalActionParams Params;
  MultiShiftFunction HeatBath;
  MultiShiftFunction Action;
  MultiShiftFunction Deriv;
  bool ActionIsNorm = false;
};

// MILC's measurements (ks_imp_rhmc control.c): after the warms, every traj_between_meas
// trajectories, the gauge observables and then npbp_reps sources for each mass in order
struct MeasurementSpec {
  int Warmups = 0;
  int Interval = 1;
  int Sources = 1;
  std::vector<ChiralCondensateParameters> Condensates;
};

struct Deck {
  HMCparameters HMC;
  CheckpointerParameters Checkpoint;
  bool ILDGCheckpoint = false;        // else Nersc
  RNGModuleParameters RNG;
  int FermionStepMultiplicity;
  int GaugeStepMultiplicity;
  RealD BareGaugeCoupling;
  RealD TadpoleFactor;
  int SeaFlavors;                     // MILC's total_dyn_flavors in the one-loop coefficients
  std::vector<FieldSpec> Fields;
  MeasurementSpec Measurements;
};

Deck readXMLDeck(const std::string& file) {
  Serialiser Reader(file, false, "grid");

  InitializationParameters Initialization(Reader);
  HamiltonianMonteCarloParameters HamiltonianMonteCarlo(Reader);
  ActionParameters Action(Reader);
  CheckpointParameters Checkpoint(Reader);
  RandomNumberGeneratorParameters RandomNumberGenerator(Reader);
  ConjugateGradientParameters CGParams(Reader);
  RationalApproximationParameters RationalParams(Reader);

  Deck deck;
  deck.HMC.MD.MDsteps = HamiltonianMonteCarlo.MolecularDynamicsSteps;
  deck.HMC.MD.trajL = HamiltonianMonteCarlo.TrajectoryLength;
  deck.HMC.NoMetropolisUntil = HamiltonianMonteCarlo.NoMetropolisUntil;
  deck.HMC.StartingType = Initialization.StartType;
  deck.Checkpoint.config_prefix = Checkpoint.ConfigurationPrefix;
  deck.Checkpoint.rng_prefix = Checkpoint.RandomNumberGeneratorPrefix;
  deck.Checkpoint.saveInterval = Checkpoint.SaveInterval;
  deck.Checkpoint.format = Checkpoint.Format;
  deck.RNG.serial_seeds = RandomNumberGenerator.SerialSeeds;
  deck.RNG.parallel_seeds = RandomNumberGenerator.ParallelSeeds;
  deck.FermionStepMultiplicity = HamiltonianMonteCarlo.FermionActionStepMultiplicity;
  deck.GaugeStepMultiplicity = HamiltonianMonteCarlo.GaugeActionStepMultiplicity;
  deck.BareGaugeCoupling = Action.BareGaugeCoupling;
  deck.TadpoleFactor = Action.TadpoleFactor;

  // Grid's thermalization (NoMetropolisUntil) plays MILC's warms: the HMC skips it
  MeasurementParameters Measurement;
  Measurement.TrajectoriesBetweenMeasurements = 1;
  Measurement.ChiralCondensateSources = 1;
  read(Reader, "MeasurementParameters", Measurement);
  deck.Measurements.Interval = Measurement.TrajectoriesBetweenMeasurements;
  deck.Measurements.Sources = Measurement.ChiralCondensateSources;
  deck.Measurements.Condensates = std::move(Measurement.ChiralCondensate);

  // MILC's rationals file: bounds [lambda_low, lambda_high] on -D^2 for every pseudofermion,
  // order_MD for the force and order_GR = order_FA for heatbath and action.
  auto field = [&](
    RealD epsilon,
    const std::vector<DeterminantFactor>& factors,
    int actionDegree,
    int forceDegree
  ) {
    const StaggeredRationalActionParams params(
      factors[0].Flavors,
      RationalParams.LambdaLow,
      RationalParams.LambdaHigh,
      CGParams.MaxIterations,
      CGParams.ActionStoppingCondition,
      actionDegree,
      CGParams.ForceStoppingCondition,
      forceDegree,
      RationalParams.Precision,
      0
    );
    return FieldSpec{
      epsilon, params,
      combinedRationalFunction(factors, params, 8, true, false),
      combinedRationalFunction(factors, params, 4, true, true),
      combinedRationalFunction(factors, params, 4, false, true)
    };
  };

  // Five pseudofermions as in MILC's in.m013m065m838 / rationals.sample.su3_rhmc_hisq_f211:
  //   1    det(Q_l)^(2/4) det(Q_s)^(1/4) det(Q_r)^(-3/4), one rational function of -D^2
  //   2-4  det(Q_r)^(1/4) each, cancelling the regulator denominator
  //   5    det(Q_c)^(1/4) on the Naik-epsilon links (Eq. A6 uses the bare charm mass)
  const RealD ml = Action.LightFermionMass;
  const RealD ms = Action.StrangeFermionMass;
  const RealD mr = Action.RegulatorFermionMass;
  const RealD mc = Action.CharmFermionMass;
  const FieldSpec lightStrange = field(
    0.0, {{ml, 2}, {ms, 1}, {mr, -3}},
    RationalParams.LightStrangeActionDegree, RationalParams.LightStrangeForceDegree
  );
  const FieldSpec regulator = field(
    0.0, {{mr, 1}}, RationalParams.RegulatorActionDegree, RationalParams.RegulatorForceDegree
  );
  const FieldSpec charm = field(
    calcNaikEpsilon(mc), {{mc, 1}},
    RationalParams.CharmActionDegree, RationalParams.CharmForceDegree
  );
  deck.Fields = {lightStrange, regulator, regulator, regulator, charm};

  // MILC's total_dyn_flavors: the sea quarks l, s, c; the regulator determinants cancel.
  deck.SeaFlavors = 2 + 1 + 1;
  return deck;
}

// MILC's ks_imp_rhmc input deck (setup.c, readin) and rationals file (load_rhmc_params.c):
// "key value(s)" in a fixed order, "#" to the end of the line is a comment
class MILCReader {
public:
  MILCReader(const std::string& file) {
    std::cout << GridLogMessage << "Reading MILC input " << file << std::endl;
    std::ifstream in(file);
    GRID_ASSERT(in.is_open() && "cannot open MILC input");
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream ss(line.substr(0, line.find('#')));
      std::string token;
      while (ss >> token) { _tokens.push_back(token); }
    }
  }

  std::string peek() const { return _pos < _tokens.size() ? _tokens[_pos] : ""; }

  std::string word() {
    GRID_ASSERT(_pos < _tokens.size() && "unexpected end of MILC input");
    return _tokens[_pos++];
  }

  void expect(const std::string& key) {
    const std::string got = word();
    if (got != key) {
      std::cout << GridLogError << "expected " << key << ", got " << got << std::endl;
      GRID_ASSERT(0);
    }
  }

  int integer(const std::string& key) { expect(key); return std::stoi(word()); }
  RealD real(const std::string& key) { expect(key); return std::stod(word()); }

  std::vector<RealD> reals(const std::string& key, int n) {
    std::vector<RealD> values;
    expect(key);
    for (int i = 0; i < n; ++i) { values.push_back(std::stod(word())); }
    return values;
  }

private:
  std::vector<std::string> _tokens;
  size_t _pos = 0;
};

// One function of the rationals file, prod_k (x + 4 m_k^2)^(y_k/z_k) as
// res[0] + sum_k res[k]/(x + pole[k]) with pole[0] a dummy, in MILC's x = -D^2 on even sites
MultiShiftFunction readMILCFunction(MILCReader& in, const std::string& tag, RealD tolerance) {
  in.reals("y_" + tag, 4);
  in.reals("z_" + tag, 4);
  in.reals("m_" + tag, 4);
  const int order = in.integer("order_" + tag);
  MultiShiftFunction f(order, 0.0, 0.0);
  f.order = order;
  f.norm = in.real("res_" + tag);
  for (auto& residue : f.residues) { residue = in.real("res_" + tag); }
  in.real("pole_" + tag);
  for (auto& pole : f.poles) { pole = in.real("pole_" + tag); }
  f.tolerances.assign(order, tolerance);
  return f;
}

Deck readMILCDeck(const std::string& file) {
  MILCReader in(file);
  Deck deck;

  in.integer("prompt");
  const Coordinate lattice(
    {in.integer("nx"), in.integer("ny"), in.integer("nz"), in.integer("nt")}
  );
  GRID_ASSERT(lattice == GridDefaultLatt() && "--grid must match nx ny nz nt");
  if (in.peek() == "node_geometry") { in.reals("node_geometry", 4); }
  if (in.peek() == "ionode_geometry") { in.reals("ionode_geometry", 4); }
  const int seed = in.integer("iseed");
  deck.RNG.parallel_seeds = std::to_string(seed);   // MILC rng: iseed, serial {iseed, volume}
  deck.RNG.serial_seeds = std::to_string(seed) + " " + std::to_string(
    lattice[0] * lattice[1] * lattice[2] * lattice[3]
  );
  const int nPseudo = in.integer("n_pseudo");
  in.expect("load_rhmc_params");
  std::string rationals = in.word();
  deck.BareGaugeCoupling = in.real("beta");
  const int nDyn = in.integer("n_dyn_masses");
  in.reals("dyn_mass", nDyn);
  deck.SeaFlavors = 0;
  for (RealD flavors : in.reals("dyn_flavors", nDyn)) { deck.SeaFlavors += flavors; }
  deck.TadpoleFactor = in.real("u0");
  deck.Measurements.Warmups = in.integer("warms");  // measurements follow the warms
  deck.HMC.Trajectories = deck.Measurements.Warmups + in.integer("trajecs");
  deck.HMC.NoMetropolisUntil = 0;                 // MILC's warms are Metropolis trajectories
  deck.Measurements.Interval = in.integer("traj_between_meas");
  const RealD timeStep = in.real("microcanonical_time_step");
  const int milcSteps = in.integer("steps_per_trajectory");
  GRID_ASSERT(milcSteps % 2 == 0 && "MILC 3G1F requires an even number of steps");
  deck.HMC.MD.MDsteps = milcSteps / 2;              // complete Omelyan periods
  deck.HMC.MD.trajL = timeStep * milcSteps;
  deck.FermionStepMultiplicity = 1;                 // MILC's INT_3G1F
  deck.GaugeStepMultiplicity = 3;

  // the rationals file, relative to the working directory as in MILC, else to the deck
  if (!std::ifstream(rationals) && file.rfind('/') != std::string::npos) {
    rationals = file.substr(0, file.rfind('/') + 1) + rationals;
  }
  MILCReader rat(rationals);
  GRID_ASSERT(rat.integer("n_pseudo") == nPseudo && "n_pseudo differs from rationals file");
  for (int i = 0; i < nPseudo; ++i) {
    // md, fa, gr: relative residuals
    const std::vector<RealD> residual = in.reals("cgresid_md_fa_gr", 3);
    const std::vector<RealD> iterations = in.reals("max_multicg_md_fa_gr", 3);
    in.reals("cgprec_md_fa_gr", 3);                 // double precision throughout

    FieldSpec field;
    field.NaikEpsilon = rat.real("naik_term_epsilon");
    // R = prod_f Q_f^(-Flavors_f/4): MD force; FA applies R^(1/2).
    field.Deriv = readMILCFunction(rat, "MD", residual[0]);
    field.HeatBath = readMILCFunction(rat, "GR", residual[2]); // R^(-1/2): heatbath
    field.Action = readMILCFunction(rat, "FA", residual[1]);
    field.ActionIsNorm = true;
    field.Params = StaggeredRationalActionParams(
      0,                                              // nf, lo, hi: logging only
      0.0,
      0.0,
      std::max({iterations[0], iterations[1], iterations[2]}),
      residual[1],
      field.Action.order,
      residual[0],
      field.Deriv.order,
      0,
      0
    );
    deck.Fields.push_back(std::move(field));
  }

  in.integer("prec_ff");
  const int nPbp = in.integer("number_of_pbp_masses");
  int maxIterations = 0;
  if (nPbp > 0) {
    maxIterations = in.integer("max_cg_prop");      // per restart
    maxIterations *= in.integer("max_cg_prop_restarts");
    deck.Measurements.Sources = in.integer("npbp_reps");
    in.integer("prec_pbp");                         // double precision throughout
  }
  for (int i = 0; i < nPbp; ++i) {
    ChiralCondensateParameters condensate;
    condensate.Mass = in.real("mass");
    condensate.NaikEpsilon = in.real("naik_term_epsilon");
    condensate.StoppingCondition = in.real("error_for_propagator");
    condensate.MaxIterations = maxIterations;
    if (in.real("rel_error_for_propagator") != 0.0) {
      std::cout << GridLogWarning << "rel_error_for_propagator is not supported; "
                << "pbp solves stop on error_for_propagator alone" << std::endl;
    }
    deck.Measurements.Condensates.push_back(condensate);
  }

  // Grid's checkpointer names lattices <prefix>.<trajectory>; a reloaded lattice must follow
  // that, Grid cannot read MILC's native format (ILDG and SciDAC files are read by name)
  auto ildg = [](const std::string& name) {
    return name.find("ildg") != std::string::npos || name.find("scidac") != std::string::npos;
  };

  const std::regex trajectory("\\.([0-9]+)$");
  std::string prefix;
  const std::string start = in.word();
  if (start == "fresh") { deck.HMC.StartingType = "ColdStart"; }
  else if (start == "warm") { deck.HMC.StartingType = "TepidStart"; }
  else if (start.rfind("reload_", 0) == 0) {
    const std::string name = in.word();
    std::smatch match;
    GRID_ASSERT(std::regex_search(name, match, trajectory) &&
                "reload needs <prefix>.<trajectory>");
    prefix = match.prefix();
    deck.HMC.StartingType = "CheckpointStart";
    deck.HMC.StartTrajectory = std::stoi(match[1]);
    deck.ILDGCheckpoint = ildg(name);
  } else { GRID_ASSERT(0 && "starting lattice command not supported"); }

  const std::string save = in.word();
  if (save == "forget") { deck.Checkpoint.saveInterval = std::numeric_limits<int>::max(); }
  else if (save.rfind("save_", 0) == 0) {
    // MILC continues as reload lat.100, save lat.110: Grid appends the trajectory itself
    prefix = std::regex_replace(in.word(), trajectory, "");
    deck.ILDGCheckpoint = ildg(save);
  } else { GRID_ASSERT(0 && "saving lattice command not supported"); }

  deck.Checkpoint.config_prefix = prefix;
  deck.Checkpoint.rng_prefix = prefix + ".rng";
  return deck;
}

////////////////////////////////////////////////////////////////
// MILC measurements: g_measure, then f_meas_imp_field per pbp mass
////////////////////////////////////////////////////////////////

class MILCGaugeAction: public PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl> {
  using Base = PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl>;
public:
  bool report = false;
  const std::array<RealD, 3> coefficients;

  MILCGaugeAction(GridCartesian* grid, RealD beta, RealD rectangle, RealD parallelogram)
    : Base(grid, beta, 1.0, rectangle, parallelogram), coefficients{1.0, rectangle, parallelogram} {}

  RealD S(const GaugeField& U) override {
    if (report) std::cout << GridLogMessage << milcRecord(
      "PLAQUETTE ACTION: %e", -6.0*context.beta*U.Grid()->gSites()*WilsonLoops<Gimpl>::avgPlaquette(U)
    ) << std::endl;
    return Base::S(U);
  }
};

class MILCMeasurements : public HmcObservable<HMCWrapper::Field> {
public:
  typedef HMCWrapper::ImplPolicy GaugeImpl;
  typedef GaugeImpl::Field GaugeField;
  typedef GaugeImpl::LinkField GaugeLinkField;
  typedef GaugeImpl::ComplexField ComplexField;
  typedef StaggeredFermionOperator::FermionField FermionField;

private:
  const MeasurementSpec _spec;
  const bool _milcOutput;
  const std::array<RealD, 3> _gaugeCoefficients;
  const int _warmupsEnd;
  std::deque<StaggeredFermionOperator> _operators;
  std::vector<ActionContract<GaugeField>> _contracts;

public:
  MILCMeasurements(
    const MeasurementSpec& spec,
    const HMCparameters& hmc,                     // after the command line
    HISQConfiguration<GaugeImpl>& links,
    GridCartesian* grid,
    GridRedBlackCartesian* rbGrid,
    const StaggeredFermionOperator::ImplParams& params,
    bool milcOutput,
    const std::array<RealD, 3>& gaugeCoefficients
  ): _spec(spec), _milcOutput(milcOutput), _gaugeCoefficients(gaugeCoefficients),
     _warmupsEnd(hmc.StartTrajectory + hmc.NoMetropolisUntil + spec.Warmups) {
    GRID_ASSERT(spec.Interval > 0 && spec.Sources > 0);
    for (const auto& c : spec.Condensates) {
      GRID_ASSERT(c.MaxIterations > 0 && "pbp solves need an iteration limit");
      // The HMC operators with the mass term 2m restored (M = 2m + D, as MILC)
      _operators.emplace_back(
        *grid, *rbGrid, 2.0 * c.Mass, 2.0, -(1.0 + c.NaikEpsilon) / 12.0, 1.0, params
      );
      _contracts.emplace_back(_operators.back().identity(), links.promise(c.NaikEpsilon));
    }
  }

  // HMC passes the trajectory just completed (Grid counts from StartTrajectory). MILC measures
  // after its warms at traj_done % traj_between_meas == traj_between_meas - 1, traj_done
  // counting measuring trajectories from 0.
  void TrajectoryComplete(
    int traj,
    ConfigurationBase<GaugeField>& links,
    GridSerialRNG& sRNG,
    GridParallelRNG& pRNG
  ) override {
    const int measuring = traj - _warmupsEnd;
    if (measuring <= 0 || measuring % _spec.Interval != 0) { return; }
    const double gaugeStart = _milcOutput ? usecond() : 0.0;
    gauge(links.get_U(false));
    if (_milcOutput) std::cout << GridLogMessage << milcRecord(
      "Aggregate time to do gauge measurement %e", (usecond() - gaugeStart) * 1.0e-6
    ) << std::endl;
    const double pbpStart = _milcOutput ? usecond() : 0.0;
    for (int i = 0; i < _operators.size(); ++i) {
      _operators[i].ImportGauge(_contracts[i]);   // HISQ links smeared by the HMC
      condensate(_spec.Condensates[i], _operators[i], pRNG);
    }
    if (_milcOutput) std::cout << GridLogMessage << milcRecord(
      "Aggregate time to do pbp measurements %e", (usecond() - pbpStart) * 1.0e-6
    ) << std::endl;
  }

  void TrajectoryComplete(int, GaugeField&, GridSerialRNG&, GridParallelRNG&) override
  { GRID_ASSERT(0 && "MILC measurements need the HISQ configuration"); }

private:
  // g_measure, KS phases out: the space-space and space-time plaquettes, Re tr per plane
  // (3 for unit links), and the Polyakov loop sum_x tr prod_t U_t(x, t) / (nx ny nz)
  void gauge(const GaugeField& U) const {
    GridBase* grid = U.Grid();
    const RealD volume = grid->gSites();

    std::vector<GaugeLinkField> link(Nd, GaugeLinkField(grid));
    for (int mu = 0; mu < Nd; ++mu) { link[mu] = PeekIndex<LorentzIndex>(U, mu); }

    ComplexField plaquette(grid);
    RealD spatial = 0.0, temporal = 0.0;
    for (int mu = 1; mu < Nd; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        WilsonLoops<GaugeImpl>::traceDirPlaquette(plaquette, link, mu, nu);
        (mu == Tdir ? temporal : spatial) += real(TensorRemove(sum(plaquette)));
    } }

    // Every time slice repeats the spatial sum, so the 4D sum over volume is MILC's average
    GaugeLinkField loop = link[Tdir];
    for (int t = 1; t < grid->GlobalDimensions()[Tdir]; ++t) {
      loop = GaugeImpl::CovShiftForward(link[Tdir], Tdir, loop);
    }
    const ComplexD polyakov = TensorRemove(sum(trace(loop))) / volume;

    std::cout << GridLogMessage << milcRecord(
      "PLAQ:\t%.16f\t%.16f", spatial / (3.0 * volume), temporal / (3.0 * volume)
    ) << std::endl;
    std::cout << GridLogMessage << milcRecord(
      "P_LOOP:\t%e\t%e", polyakov.real(), polyakov.imag()
    ) << std::endl;
    if (!_milcOutput) return;

    // MILC's Symanzik loop table: six plaquettes, twelve rectangles, sixteen parallelograms.
    // Its backward direction is 7-mu; links here have no staggered or boundary phases.
    double action = 0.0;
    GaugeLinkField product(grid);
    auto measure = [&](int family, int index, std::initializer_list<int> path) {
      product = 1.0;
      for (auto step = std::rbegin(path); step != std::rend(path); ++step) {
        const int dir = *step, mu = dir < Nd ? dir : 7-dir;
        if (dir < Nd) product = GaugeImpl::CovShiftForward(link[mu], mu, product);
        else product = GaugeImpl::CovShiftBackward(link[mu], mu, product);
      }
      const double average = real(TensorRemove(sum(trace(product)))) / volume;
      action += _gaugeCoefficients[family]*(3.0-average);
      std::string record = milcRecord("G_LOOP:  %d  %d  %d   \t%.16e\t( ", family, index, int(path.size()), average);
      for (int dir : path) record += std::to_string(dir) + " ";
      std::cout << GridLogMessage << record << " )" << std::endl;
    };
    int index = 0;
    for (int mu = 0; mu < Nd; ++mu) for (int nu = mu+1; nu < Nd; ++nu)
      measure(0, index++, {mu, nu, 7-mu, 7-nu});
    index = 0;
    for (int mu = 0; mu < Nd; ++mu) for (int nu = 0; nu < Nd; ++nu) if (nu != mu)
      measure(1, index++, {mu, mu, nu, 7-mu, 7-mu, 7-nu});
    index = 0;
    for (int mu = 0; mu < Nd; ++mu) for (int nu = mu+1; nu < Nd; ++nu)
      for (int rho = nu+1; rho < Nd; ++rho) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c) {
        const int v = b ? 7-nu : nu, w = c ? 7-rho : rho;
        measure(2, index++, {mu, v, w, 7-mu, 7-v, 7-w});
      }
    std::cout << GridLogMessage << milcRecord("GACTION: %e", action) << std::endl;
  }

  // f_meas_imp_field: per source eta on all sites (grsource_plain_field, drawn as the HMC
  // heatbath draws), pbp = (2/V) eta^dag M^-1 eta on even and odd sites separately and
  // FACTION = (1/V) Re (M^dag eta)^dag M^-1 eta on even sites
  void condensate(
    const ChiralCondensateParameters& c,
    StaggeredFermionOperator& op,
    GridParallelRNG& pRNG
  ) const {
    GridBase* grid = op.FermionGrid();
    GridBase* rbGrid = op.FermionRedBlackGrid();
    const RealD volume = grid->gSites();

    FermionField source(grid), image(grid), solution(grid);
    FermionField left(rbGrid), right(rbGrid);
    ConjugateGradient<FermionField> CG(c.StoppingCondition, c.MaxIterations, false);
    SchurRedBlackStaggeredSolve<FermionField> solve(CG);

    for (int rep = 0; rep < _spec.Sources; ++rep) {
      gaussian(pRNG, source);
      source *= std::sqrt(0.5);                   // MILC's exp(-x^2) per component
      op.Mdag(source, image);                     // M^dag eta = 2m eta - D eta
      solution = Zero();
      solve(op, source, solution);

      pickCheckerboard(Even, left, source);
      pickCheckerboard(Even, right, solution);
      const ComplexD even = innerProduct(left, right);
      pickCheckerboard(Even, left, image);
      const RealD action = real(innerProduct(left, right));
      pickCheckerboard(Odd, left, source);
      pickCheckerboard(Odd, right, solution);
      const ComplexD odd = innerProduct(left, right);

      std::cout << GridLogMessage << milcRecord(
        "PBP: mass %e     %e  %e  %e  %e ( %d of %d )", c.Mass,
        2.0 * even.real() / volume, 2.0 * odd.real() / volume,
        2.0 * even.imag() / volume, 2.0 * odd.imag() / volume,
        rep + 1, _spec.Sources
      ) << std::endl;
      std::cout << GridLogMessage << milcRecord(
        "FACTION: mass = %e,  %e ( %d of %d )", c.Mass, action / volume, rep + 1, _spec.Sources
      ) << std::endl;
    }
  }
};

// AddObservable takes an HMC module owning the observable
class MILCMeasurementModule : public HMCModuleBase<HmcObservable<HMCWrapper::Field>> {
  MILCMeasurements _measurements;

public:
  template <class... Args>
  explicit MILCMeasurementModule(Args&&... args): _measurements(std::forward<Args>(args)...) {}

  HmcObservable<HMCWrapper::Field>* getPtr() override { return &_measurements; }
};

////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  const double setupStart = usecond();
  Grid_init(&argc, &argv);

  ////////////////////////////////////////////////////////////////
  // Input: --xml <file>, or --milc-input <deck> with MILC's input deck and rationals file
  ////////////////////////////////////////////////////////////////
  const bool milc = GridCmdOptionExists(argv, argv + argc, "--milc-input");
  Deck deck = milc ? readMILCDeck(GridCmdOptionPayload(argv, argv + argc, "--milc-input"))
                   : readXMLDeck(GridCmdOptionPayload(argv, argv + argc, "--xml"));

  /////////////////////////////////////////////////////////////////
  // Hamiltonian Monte Carlo wrapper
  /////////////////////////////////////////////////////////////////
  deck.HMC.MD.name = "InterleavedIntegrator";
  HMCWrapper TheHMC(deck.HMC);

  TheHMC.ReadCommandLine(argc, argv);
  TheHMC.Resources.AddFourDimGrid("gauge");
  if (deck.ILDGCheckpoint) { TheHMC.Resources.LoadILDGCheckpointer(deck.Checkpoint); }
  else { TheHMC.Resources.LoadNerscCheckpointer(deck.Checkpoint); }
  TheHMC.Resources.SetRNGSeeds(deck.RNG);

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();

  ////////////////////////////////////////////////////////////////
  // Fermion operators: one per Naik epsilon
  ////////////////////////////////////////////////////////////////
  std::vector<Complex> boundary = {1, 1, 1, -1};
  StaggeredFermionOperator::ImplParams Params(boundary);
  HISQConfiguration<HMCWrapper::ImplPolicy> Policy(GridPtr);

  // Match MILC: M = 2m + Dslash, so double Grid's hopping coefficients. This also
  // matches MILC pseudofermion normalization. As in MILC the operators are massless:
  // the rational functions are of -D^2, with each bare mass in their poles as (2m)^2.
  std::map<RealD, StaggeredFermionOperator> Operators;

  ////////////////////////////////////////////////////////////////
  // Full action
  ////////////////////////////////////////////////////////////////
  ActionLevel<HMCWrapper::Field> Level1(deck.FermionStepMultiplicity);
  ActionLevel<HMCWrapper::Field> Level2(deck.GaugeStepMultiplicity);

  std::deque<FermionAction> Fields;
  HISQPseudoFermions Fermions; // all fields share one HISQ pullback per force
  for (const auto& spec : deck.Fields) {
    auto& Op = Operators.try_emplace(
      spec.NaikEpsilon,
      *GridPtr, *GridRBPtr, 0.0, 2.0, -(1.0 + spec.NaikEpsilon) / 12.0, 1.0, Params
    ).first->second;
    Fields.emplace_back(Op, spec.Params, spec.HeatBath, spec.Action, spec.Deriv, spec.ActionIsNorm);
    Fields.back().signContract(Policy.promise(spec.NaikEpsilon));
    Fermions.push_back(&Fields.back());
  }
  Level1.push_back(&Fermions);

  // Eq. A2, beta = 10/g^2. Tadpole improvement is gauge-only.
  const RealD u0 = deck.TadpoleFactor;
  const int nf = deck.SeaFlavors;
  MILCGaugeAction GaugeAction(
    GridPtr,
    deck.BareGaugeCoupling,
    oneLoopMILCRectangleCoefficient(u0, nf), 
    oneLoopMILCParallelogramCoefficient(u0, nf)
  );
  Level2.push_back(&GaugeAction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);

  //////////////////////////////////////////////////////////////
  // Run HMC with HISQ smearing
  //////////////////////////////////////////////////////////////
  const bool milcOutputMixed = GridCmdOptionExists(argv, argv + argc, "--milc-output-mixed");
  const bool milcOutput = milcOutputMixed || GridCmdOptionExists(argv, argv + argc, "--milc-output");
  
  MILCGaugeImpl::report = milcOutput;
  GaugeAction.report = milcOutput;
  const bool gaugePerformance = milcOutput || GridLogPerformance.isActive();
  Policy.setPerformanceReporting(gaugePerformance);
  
  TheHMC.Resources.AddObservable<PlaqObs>();
  TheHMC.Resources.AddObservable<MILCMeasurementModule>(
    deck.Measurements, TheHMC.Parameters, Policy, GridPtr, GridRBPtr, Params, milcOutput, GaugeAction.coefficients
  );

  {
    // Only redirect the printing rank: rdbuf() clears Grid's MPI quiet state.
    MILCOutput output(
      milcOutput && GridPtr->IsBoss(), 
      Level1.actions.size(), // fermion actions as the HMC sees them (one group)
      GridPtr->gSites(),
      TheHMC.Parameters.MD.MDsteps * (milc ? 2 : 1),
      TheHMC.Parameters.StartTrajectory,
      TheHMC.Parameters.NoMetropolisUntil + deck.Measurements.Warmups,
      deck.Measurements.Interval, 
      GridPtr->ProcessorCount(), 
      !milcOutputMixed
    );
    
    TheHMC.Run(Policy, [&](auto& md) {
      for (int level = 0; level < md.levels; ++level)
      { md.schemes[level] = omelyan2(md.T, md.V[level], md.as[level].multiplier, 0.2); }
      md.evolution.set_branches({&md.schemes[0], &md.schemes[1]});
      
      if (gaugePerformance) {
        double actionWork = double(GaugeAction.grid->lSites())*
          GaugeAction.actionFlopsPerSite(GaugeAction.context) + 2*(GridPtr->lSites()-1) + 1;
        
        if (milcOutput) { // S also evaluates the opt-in PLAQUETTE ACTION diagnostic on the tight grid
          using M = MatrixMath::DenseMatrixFlopModel<Nc>;
          actionWork += double(GridPtr->lSites())*(Nd*(Nd-1)/2)*(3*M::multiply + M::trace + 2)
            + 2*(GridPtr->lSites()-1) + 4;
        }
        
        const double forceWork = double(GaugeAction.grid->lSites())
          *GaugeAction.forceFlopsPerSite(GaugeAction.context);
        
        std::cout << GridLogPerformance << milcRecord(
          "GaugeWorkModel: action = %.17e force = %.17e", actionWork, forceWork
        ) << std::endl;
      }
      
      if (milcOutput) std::cout << GridLogMessage << milcRecord(
        "Aggregate time to setup %e", (usecond() - setupStart) * 1.0e-6
      ) << std::endl;
    });

    if (gaugePerformance) {
      auto report = UnitaryProjectionReport::read(false); // one read at run end, no per-stage transfers
      GridPtr->GlobalSum(report.svd); GridPtr->GlobalSum(report.filter);
      std::cout << GridLogMessage << "hisq_svd_counter = " << report.svd << std::endl;
      std::cout << GridLogMessage << "hisq_force_filter_counter = " << report.filter << std::endl;
    }
    output.complete();
  }

  Grid_finalize();
}

/* MILC input (pass as --milc-input <deck>, with --grid equal to nx ny nz nt): the ks_imp_rhmc deck
   of su3_rhmc_hisq (ks_imp_rhmc/test/su3_rhmc_hisq.f211.1.sample-in) and the rationals file it
   names in load_rhmc_params. Each pseudofermion uses the massless operator with the
   naik_term_epsilon of the file; GR is the heatbath, MD the force, and FA the
   half-power used to evaluate the action as |FA Phi|^2; cgresid_md_fa_gr are the
   relative residuals of the multishift solves, max_multicg_md_fa_gr their iteration limits;
   warms + trajecs trajectories with the Metropolis test throughout; MILC's
   steps_per_trajectory/2 complete Omelyan periods of 2*microcanonical_time_step,
   lambda = 0.2, and independent fermion/gauge multiplicities 1 and 3. XML counts
   complete periods directly; TrajectoryLength/MolecularDynamicsSteps is their length;
   iseed seeds the MILC rng (serial {iseed, volume}); dyn_flavors give the one-loop gauge
   coefficients. fresh is ColdStart, warm TepidStart; reload_* needs a lattice named
   <prefix>.<trajectory> that Grid's checkpointer can read (the ILDG checkpointer for names
   containing ildg or scidac, MILC's native format cannot be read); save_* writes Grid's
   checkpoints every trajectory under the given name, less any .<trajectory> ending, forget
   writes nothing. One checkpointer both reads and writes, so a continuation keeps its prefix
   (reload_serial lat.100, save_serial lat.110). After the warms, every traj_between_meas
   trajectories print PLAQ and P_LOOP and, for each pbp mass in order, npbp_reps Gaussian
   sources on all sites give PBP and FACTION, drawn from the parallel rng after the trajectory
   as in MILC; each mass solves 2m + D on its naik_term_epsilon links to error_for_propagator
   in at most max_cg_prop * max_cg_prop_restarts iterations (a nonzero
   rel_error_for_propagator is ignored with a warning). cgprec_md_fa_gr, prec_ff and prec_pbp
   are read and ignored (double precision throughout); continue is not supported.
*/

/* Example XML input file (pass as --xml <xml-file-name>.xml):
   The masses, beta, and u0 are the a ~ 0.12 fm physical-mass ensemble in
   Table IV of Phys. Rev. D 87, 054505. The regulator mass is 0.2 as in
   Phys. Rev. D 82, 074501. Step counts are examples, not a reproduction of a MILC
   production input deck; check them before use. The rational bounds, the regulator and
   charm orders and the rng seeds follow in.m013m065m838 (1e-15, 90, orders 7/9). The
   combined field approximates each of its three factors separately at the given degree
   (3 x degree poles); its light factor x^(2/4) on [4 m_l^2, 90] reaches 1e-9 at degree 20
   with 80 digits. With --enable-rng=milc the parallel rng takes one seed and the serial rng
   {seed, volume} (4096 below is 8^4). MeasurementParameters is optional (the gauge
   observables every trajectory, no condensate): measurements begin after the
   NoMetropolisUntil trajectories, and each ChiralCondensate elem is one MILC pbp mass.
<?xml version="1.0"?>
<grid>
  <InitializationParameters>
    <StartType>TepidStart</StartType>
  </InitializationParameters>
  <HamiltonianMonteCarloParameters>
    <NoMetropolisUntil>0</NoMetropolisUntil>
    <TrajectoryLength>1.0</TrajectoryLength>
    <MolecularDynamicsSteps>15</MolecularDynamicsSteps>
    <FermionActionStepMultiplicity>1</FermionActionStepMultiplicity>
    <GaugeActionStepMultiplicity>3</GaugeActionStepMultiplicity>
  </HamiltonianMonteCarloParameters>
  <ActionParameters>
    <BareGaugeCoupling>6.0</BareGaugeCoupling>
    <TadpoleFactor>0.86372</TadpoleFactor>
    <LightFermionMass>0.00184</LightFermionMass>
    <StrangeFermionMass>0.0507</StrangeFermionMass>
    <CharmFermionMass>0.628</CharmFermionMass>
    <RegulatorFermionMass>0.2</RegulatorFermionMass>
  </ActionParameters>
  <CheckpointParameters>
    <Format>IEEE64BIG</Format>
    <ConfigurationPrefix>ckpoint_hmc_lat</ConfigurationPrefix>
    <RandomNumberGeneratorPrefix>ckpoint_hmc_rng</RandomNumberGeneratorPrefix>
    <SaveInterval>1</SaveInterval>
  </CheckpointParameters>
  <RandomNumberGeneratorParameters>
    <SerialSeeds>6 4096</SerialSeeds>
    <ParallelSeeds>6</ParallelSeeds>
  </RandomNumberGeneratorParameters>
  <ConjugateGradientParameters>
    <ActionStoppingCondition>1e-10</ActionStoppingCondition>
    <ForceStoppingCondition>1e-8</ForceStoppingCondition>
    <MaxIterations>10000</MaxIterations>
  </ConjugateGradientParameters>
  <RationalApproximationParameters>
    <LambdaLow>1e-15</LambdaLow>
    <LambdaHigh>90.0</LambdaHigh>
    <Precision>80</Precision>
    <LightStrangeActionDegree>20</LightStrangeActionDegree>
    <LightStrangeForceDegree>16</LightStrangeForceDegree>
    <RegulatorActionDegree>9</RegulatorActionDegree>
    <RegulatorForceDegree>7</RegulatorForceDegree>
    <CharmActionDegree>9</CharmActionDegree>
    <CharmForceDegree>7</CharmForceDegree>
  </RationalApproximationParameters>
  <MeasurementParameters>
    <TrajectoriesBetweenMeasurements>1</TrajectoriesBetweenMeasurements>
    <ChiralCondensateSources>1</ChiralCondensateSources>
    <ChiralCondensate>
      <elem>
        <Mass>0.00184</Mass>
        <NaikEpsilon>0.0</NaikEpsilon>
        <StoppingCondition>1e-8</StoppingCondition>
        <MaxIterations>10000</MaxIterations>
      </elem>
    </ChiralCondensate>
  </MeasurementParameters>
</grid>
*/
