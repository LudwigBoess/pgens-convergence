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
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include "kernels/particle_moments.hpp"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct ConstDensity : public arch::SpatialDistribution<S, M> {
    ConstDensity(const M& metric)
      : arch::SpatialDistribution<S, M> { metric } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      return ONE;
    }
  };

  template <SimEngine::type S, class M>
  struct MaxwellWave : public arch::EnergyDistribution<S, M> {
    
    MaxwellWave(const M& metric, random_number_pool_t& pool, real_t temp, real_t A_wave, real_t lambda_L, real_t sign) 
        : arch::EnergyDistribution<S, M>{metric}
        , pool {pool} 
        , temp { temp } 
        , A_wave { A_wave }
        , lambda_L { lambda_L }
        , sign { sign }  {}

    Inline void operator()(const coord_t<M::Dim>& x_Ph, vec_t<Dim::_3D>& v) const {
      arch::JuttnerSinge(v, temp, pool);
      v[0] += sign * math::sqrt(A_wave) / constant::PI * lambda_L * math::cos(constant::TWO_PI * x_Ph[0] / lambda_L);
    }

  private:
    random_number_pool_t pool;
    real_t temp, A_wave, lambda_L, sign;
  };


  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions { traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    Metadomain<S, M>& global_domain;

    // domain properties
    const real_t global_size;
    // gas properties
    const real_t temperature;
    // wave properties
    const real_t wave_amplitude, wave_mode;
    // derived properties
    const real_t ppc0;
    const ncells_t ncells0;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_size { global_domain.mesh().extent(in::x1).second - global_domain.mesh().extent(in::x1).first }
      , ncells0 { global_domain.mesh().n_all(in::x1) }
      , temperature { p.template get<real_t>("setup.temperature") }
      , wave_amplitude { p.template get<real_t>("setup.wave_amplitude") }
      , wave_mode { p.template get<real_t>("setup.wave_mode") }
      , ppc0 { p.template get<real_t>("particles.ppc0") }
      {}

    inline PGen() {}

    inline void InitPrtls(Domain<S, M>& domain) {

      const auto lambda_L = global_size / wave_mode;

      // define cold maxwellian
      const auto T_e = temperature / domain.species[0].mass();
      const auto maxwellian_e = MaxwellWave<S, M>( domain.mesh.metric, domain.random_pool(), 
                                                  T_e, wave_amplitude, lambda_L, ONE);

      const auto T_p = temperature / domain.species[1].mass();
      const auto maxwellian_p = MaxwellWave<S, M>( domain.mesh.metric, domain.random_pool(), 
                                                  T_p, wave_amplitude, lambda_L, -ONE);
      // define density step
      
      const auto density_const = ConstDensity<S, M>(domain.mesh.metric);

      // inject particles with a density step and a maxwellian energy distribution
      arch::InjectNonUniform<S, M, decltype(maxwellian_e), decltype(maxwellian_p), decltype(density_const)>(
        params,
        domain,
        { 1, 2 },
        { maxwellian_e, maxwellian_p },
        density_const,
        ONE);

    }

  };
} // namespace user
#endif
