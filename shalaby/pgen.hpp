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
#include "archetypes/spatial_dist.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"


namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t wave_amplitude, real_t lambda)
      : wave_amplitude { wave_amplitude }
      , lambda { lambda } {}

    // electric field components
    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return -wave_amplitude * math::sin(constant::TWO_PI * x_Ph[0] / lambda);
    }

  private:
    const real_t wave_amplitude, lambda;
  };

    template <SimEngine::type S, class M>
  struct DensityWave : public arch::SpatialDistribution<S, M> {
    DensityWave(const M& metric, real_t lambda)
      : arch::SpatialDistribution<S, M> { metric }
      , lambda { lambda } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
        return HALF * (ONE + math::cos(constant::TWO_PI * x_Ph[0] / lambda));
    }

  private:
    const real_t lambda;
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

    Metadomain<S, M>& global_domain;

    // domain properties
    const real_t global_size;
    // gas properties
    const real_t temperature;
    // wave properties
    const real_t wave_amplitude, wave_mode, lambda_L;

    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , global_size { global_domain.mesh().extent(in::x1).second - global_domain.mesh().extent(in::x1).first }
      , temperature { p.template get<real_t>("setup.temperature") }
      , wave_amplitude { p.template get<real_t>("setup.wave_amplitude") }
      , wave_mode { p.template get<real_t>("setup.wave_mode") }
      , lambda_L { global_size / wave_mode }
      , init_flds { wave_amplitude, lambda_L } {}

    inline PGen() {}

    inline void InitPrtls(Domain<S, M>& domain) {

      // define cold maxwellian
      const auto T_e = temperature / domain.species[0].mass();
      const auto maxwellian = arch::Maxwellian<S, M>( domain.mesh.metric, domain.random_pool(), T_e);

      // define particle distribution
      const auto density_wave = DensityWave<S, M>(domain.mesh.metric, lambda_L);
      const auto density_value = FOUR * constant::PI * params.template get<real_t>("scales.skindepth0") * wave_amplitude;

      // inject particles with a density step and a maxwellian energy distribution
      arch::InjectNonUniform<S, M, decltype(maxwellian), decltype(maxwellian), decltype(density_wave)>(
        params,
        domain,
        { 1, 2 },
        { maxwellian, maxwellian },
        density_wave,
        density_value);

        // set ions to zero to save on computational cost
        domain.species[1].set_npart(0);

    }

  };
} // namespace user
#endif
