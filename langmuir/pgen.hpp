#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct SinEDist : public arch::EnergyDistribution<S, M> {
    SinEDist(const M&                   metric,
             real_t                     v_max,
             real_t                     temperature,
             random_number_pool_t&      pool,
             const std::vector<int>&    n,
             const std::vector<real_t>& s)
      : arch::EnergyDistribution<S, M> { metric }
      , v_max { v_max }
      , temperature { temperature }
      , pool { pool }
      , kx1 { s.size() > 0 ? static_cast<real_t>(constant::TWO_PI) *
                               static_cast<real_t>(n[0]) / s[0]
                           : ZERO }
      , kx2 { s.size() > 1 ? static_cast<real_t>(constant::TWO_PI) *
                               static_cast<real_t>(n[1]) / s[1]
                           : ZERO }
      , kx3 { s.size() > 2 ? static_cast<real_t>(constant::TWO_PI) *
                               static_cast<real_t>(n[2]) / s[2]
                           : ZERO } {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph,
                           vec_t<Dim::_3D>&       v) const {

        // sample from a Maxwellian distribution with the given temperature, then modulate the velocity by a sinusoidal function of position
        arch::JuttnerSinge(v, temperature, pool);
        const auto k = math::sqrt(SQR(kx1) + SQR(kx2) + SQR(kx3));
        if constexpr (M::Dim == Dim::_1D) {
          v[0] += v_max * math::sin(kx1 * x_Ph[0]);
        } else if constexpr (M::Dim == Dim::_2D) {
          v[0] += v_max * kx1 / k * math::sin(kx1 * x_Ph[0] + kx2 * x_Ph[1]);
          v[1] += v_max * kx2 / k * math::sin(kx1 * x_Ph[0] + kx2 * x_Ph[1]);
        } else {
          v[0] += v_max * kx1 / k *
                 math::sin(kx1 * x_Ph[0] + kx2 * x_Ph[1] + kx3 * x_Ph[2]);
          v[1] += v_max * kx2 / k *
                 math::sin(kx1 * x_Ph[0] + kx2 * x_Ph[1] + kx3 * x_Ph[2]);
          v[2] += v_max * kx3 / k *
                 math::sin(kx1 * x_Ph[0] + kx2 * x_Ph[1] + kx3 * x_Ph[2]);
        }
    }

  private:
    random_number_pool_t pool;
    const real_t v_max, temperature, kx1, kx2, kx3;
  };

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

    const real_t sx1, sx2, sx3;
    const real_t vmax, temperature;
    const int    nx1, nx2, nx3;

    std::vector<real_t> svec;
    std::vector<int>    nvec;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , sx1 { global_domain.mesh().extent(in::x1).second -
              global_domain.mesh().extent(in::x1).first }
      , sx2 { global_domain.mesh().extent(in::x2).second -
              global_domain.mesh().extent(in::x2).first }
      , sx3 { global_domain.mesh().extent(in::x3).second -
              global_domain.mesh().extent(in::x3).first }
      , temperature { p.get<real_t>("setup.temperature", 0.001) }
      , vmax { p.get<real_t>("setup.vmax", 0.01) }
      , nx1 { p.get<int>("setup.nx1", 10) }
      , nx2 { p.get<int>("setup.nx2", 10) }
      , nx3 { p.get<int>("setup.nx3", 10) } {
      const auto sxs = std::vector<real_t> { sx1, sx2, sx3 };
      const auto nxs = std::vector<int> { nx1, nx2, nx3 };
      for (auto d = 0u; d < M::Dim; ++d) {
        svec.push_back(sxs[d]);
        nvec.push_back(nxs[d]);
      }
    }

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = SinEDist<S, M>(local_domain.mesh.metric,
                                              vmax, temperature,
                                              local_domain.random_pool(),
                                              nvec,
                                              svec);
      const auto maxwellian   = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                     local_domain.random_pool(),
                                                     temperature);
      arch::InjectUniform<S, M, decltype(energy_dist), decltype(maxwellian)>(params,
                                                                       local_domain,
                                                                       { 1, 2 },
                                                                       {energy_dist, maxwellian},
                                                                       ONE);
    }
  };

} // namespace user

#endif
