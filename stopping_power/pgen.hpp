#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    using prmvec_t = std::vector<real_t>;
    const Metadomain<S, M>& global_domain;

    prmvec_t drifts_in_x, drifts_in_y, drifts_in_z;
    prmvec_t densities, temperatures;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , drifts_in_x { p.template get<prmvec_t>("setup.drifts_in_x", prmvec_t {}) }
      , drifts_in_y { p.template get<prmvec_t>("setup.drifts_in_y", prmvec_t {}) }
      , drifts_in_z { p.template get<prmvec_t>("setup.drifts_in_z", prmvec_t {}) }
      , densities { p.template get<prmvec_t>("setup.densities", prmvec_t {}) }
      , temperatures { p.template get<prmvec_t>("setup.temperatures", prmvec_t {}) } {
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
      const auto nspec = domain.species.size() - 2;
      const auto drift_1  = prmvec_t { drifts_in_x[0],
                                      drifts_in_y[0],
                                      drifts_in_z[0] };
      const auto drift_2  = prmvec_t { drifts_in_x[1],
                                      drifts_in_y[1],
                                      drifts_in_z[1] };
      const auto injector = arch::experimental::
        UniformInjector<S, M, arch::experimental::Maxwellian, arch::experimental::Maxwellian>(
          arch::experimental::Maxwellian<S, M>(domain.mesh.metric,
                                               domain.random_pool,
                                               domain.species[1].mass() * temperatures[0],
                                               drift_1),
          arch::experimental::Maxwellian<S, M>(domain.mesh.metric,
                                               domain.random_pool,
                                               temperatures[1],
                                               drift_2),
          { 1, 2 });
      arch::experimental::InjectUniform<S, M, decltype(injector)>(
        params,
        domain,
        injector,
        densities[0]);
      

      const auto empty = std::vector<real_t> {};
      const auto x1_e  = params.template get<std::vector<real_t>>("setup.x1_e",
                                                                 empty);
      const auto x2_e  = params.template get<std::vector<real_t>>("setup.x2_e",
                                                                 empty);
      const auto x3_e  = params.template get<std::vector<real_t>>("setup.x3_e",
                                                                 empty);
      const auto phi_e = params.template get<std::vector<real_t>>("setup.phi_e",
                                                                  empty);
      const auto ux1_e = params.template get<std::vector<real_t>>("setup.ux1_e",
                                                                  empty);
      const auto ux2_e = params.template get<std::vector<real_t>>("setup.ux2_e",
                                                                  empty);
      const auto ux3_e = params.template get<std::vector<real_t>>("setup.ux3_e",
                                                                  empty);

      const auto x1_i  = params.template get<std::vector<real_t>>("setup.x1_i",
                                                                 empty);
      const auto x2_i  = params.template get<std::vector<real_t>>("setup.x2_i",
                                                                 empty);
      const auto x3_i  = params.template get<std::vector<real_t>>("setup.x3_i",
                                                                 empty);
      const auto phi_i = params.template get<std::vector<real_t>>("setup.phi_i",
                                                                  empty);
      const auto ux1_i = params.template get<std::vector<real_t>>("setup.ux1_i",
                                                                  empty);
      const auto ux2_i = params.template get<std::vector<real_t>>("setup.ux2_i",
                                                                  empty);
      const auto ux3_i = params.template get<std::vector<real_t>>("setup.ux3_i",
                                                                  empty);
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
      } else if constexpr (D == Dim::_2D) {
        data_e["phi"] = phi_e;
        data_i["phi"] = phi_i;
      }

      arch::InjectGlobally<S, M>(global_domain, domain, (spidx_t)3, data_e);
      arch::InjectGlobally<S, M>(global_domain, domain, (spidx_t)4, data_i);

    }
  };

} // namespace user

#endif
