#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "archetypes/traits.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines =
      arch::traits::pgen::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics =
      arch::traits::pgen::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      arch::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    using prmvec_t = std::vector<real_t>;
    const Metadomain<S, M>& global_domain;

    const real_t  global_xmin, global_xmax;
    prmvec_t drifts_in_x, drifts_in_y, drifts_in_z;
    prmvec_t densities, temperatures;
    const real_t drift_ux;
    const int Nsampled;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_xmin { global_domain.mesh().extent(in::x1).first }
      , global_xmax { global_domain.mesh().extent(in::x1).second }
      , drifts_in_x { p.template get<prmvec_t>("setup.drifts_in_x", prmvec_t {}) }
      , drifts_in_y { p.template get<prmvec_t>("setup.drifts_in_y", prmvec_t {}) }
      , drifts_in_z { p.template get<prmvec_t>("setup.drifts_in_z", prmvec_t {}) }
      , densities { p.template get<prmvec_t>("setup.densities", prmvec_t {}) }
      , temperatures { p.template get<prmvec_t>("setup.temperatures", prmvec_t {}) } 
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
       , Nsampled { p.template get<int>("setup.Nsampled") } {
      const auto nspec = p.template get<std::size_t>("particles.nspec");


      for (auto* specs :
           { &drifts_in_x, &drifts_in_y, &drifts_in_z, &temperatures }) {
        if (specs->empty()) {
          for (auto n = 0u; n < nspec; ++n) {
            specs->push_back(ZERO);
          }
        }
      }
      if (densities.empty()) {
        for (auto n = 0u; n < nspec; n += 2) {
          densities.push_back(TWO / static_cast<real_t>(nspec));
        }
      }
    }

    inline void InitPrtls(Domain<S, M>& domain) {

      // background plasma
      const auto drift_1 = prmvec_t { drifts_in_x[0],
                                        drifts_in_y[0],
                                        drifts_in_z[0] };
        const auto drift_2 = prmvec_t { drifts_in_x[1],
                                        drifts_in_y[1],
                                        drifts_in_z[1] };
      arch::InjectUniformMaxwellians<S, M>(params, domain, 
            ONE, { temperatures[0], temperatures[1] }, { 1, 2 },
            { drift_1, drift_2 });

      Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(42);

      // injected particles
      for (int n = 0; n < Nsampled; ++n) {
        auto rand_gen = pool.get_state();
        auto rand_X1  = static_cast<real_t>(rand_gen.drand(0., 1.));
        
        const auto x1_e  = {global_xmin + static_cast<real_t>(rand_gen.drand(0., 1.)) * (global_xmax - global_xmin)};
        const auto x1_i  = x1_e;

        const auto x2_e  = {global_xmin + static_cast<real_t>(rand_gen.drand(0., 1.)) * (global_xmax - global_xmin)};
        const auto x2_i  = x1_e;

        const auto x3_e  = {global_xmin + static_cast<real_t>(rand_gen.drand(0., 1.)) * (global_xmax - global_xmin)};
        const auto x3_i  = x1_e;

        // electron velocities
        const auto vx_e = static_cast<real_t>(rand_gen.drand(0., 1.));
        const auto vy_e = static_cast<real_t>(rand_gen.drand(0., 1.));
        const auto vz_e = static_cast<real_t>(rand_gen.drand(0., 1.));
        const auto v_norm_e = math::sqrt(SQR(vx_e) + SQR(vy_e) + SQR(vz_e));
        const auto ux1_e = {drift_ux * vx_e / v_norm_e};
        const auto ux2_e = {drift_ux * vy_e / v_norm_e};
        const auto ux3_e = {drift_ux * vz_e / v_norm_e};

        // ion velocities
        const auto vx_i = static_cast<real_t>(rand_gen.drand(0., 1.));
        const auto vy_i = static_cast<real_t>(rand_gen.drand(0., 1.));
        const auto vz_i = static_cast<real_t>(rand_gen.drand(0., 1.));
        const auto v_norm_i = math::sqrt(SQR(vx_i) + SQR(vy_i) + SQR(vz_i));
        const auto ux1_i = {drift_ux * vx_i / v_norm_i};
        const auto ux2_i = {drift_ux * vy_i / v_norm_i};
        const auto ux3_i = {drift_ux * vz_i / v_norm_i};

        std::map<std::string, std::vector<real_t>> data_e {
          {  "x1",  x1_e },
          {  "x2",  x2_e },
          { "ux1", ux1_e },
          { "ux2", ux2_e },
          { "ux3", ux3_e }
        };
        std::map<std::string, std::vector<real_t>> data_i {
          {  "x1",  x1_i },
          {  "x2",  x2_i },
          { "ux1", ux1_i },
          { "ux2", ux2_i },
          { "ux3", ux3_i }
        };
        if constexpr (M::CoordType == Coord::Cart or D == Dim::_3D) {
          data_e["x3"] = x3_e;
          data_i["x3"] = x3_i;
        }

        arch::InjectGlobally<S, M>(global_domain, domain, (spidx_t)3, data_e);
        arch::InjectGlobally<S, M>(global_domain, domain, (spidx_t)4, data_i);
        
        pool.free_state(rand_gen);

      }
    }
  };

} // namespace user

#endif
