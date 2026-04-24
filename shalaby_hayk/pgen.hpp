#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/traits.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t wave_E_ovr_B, real_t wave_k)
      : wave_E_ovr_B { wave_E_ovr_B }
      , wave_k { wave_k } {}

    // electric field components
    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return wave_E_ovr_B * math::sin(wave_k * x_Ph[0]);
    }

  private:
    const real_t wave_E_ovr_B, wave_k;
  };

  template <SimEngine::type S, class M>
  struct DensityWave : public arch::SpatialDistribution<S, M> {
    DensityWave(const M& metric, real_t delta_n_e, real_t wave_k)
      : arch::SpatialDistribution<S, M> { metric }
      , delta_n_e { delta_n_e }
      , wave_k { wave_k } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      return ONE - delta_n_e * math::cos(wave_k * x_Ph[0]);
    }

  private:
    const real_t delta_n_e, wave_k;
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

    // initial background temperature
    const real_t temperature;
    // Ew / B0 at a maximum of the wave: E(x) = Ew * cos(kx * x)
    const real_t wave_E_ovr_B;
    // kx = n * 2 pi / Lx
    const real_t wave_k;
    // delta_n_e = (Ew / B0) * k d0 * sqrt(sigma0)
    const real_t delta_n_electron;

    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& metadomain)
      : arch::ProblemGenerator<S, M> { p }
      , temperature { p.template get<real_t>("setup.temperature") }
      , wave_E_ovr_B { p.template get<real_t>("setup.wave_E_ovr_B") }
      , wave_k { static_cast<real_t>(constant::TWO_PI) *
                 p.template get<real_t>("setup.wave_number") /
                 (metadomain.mesh().extent(in::x1).second -
                  metadomain.mesh().extent(in::x1).first) }
      , delta_n_electron { wave_E_ovr_B * wave_k *
                           p.template get<real_t>("scales.skindepth0") *
                           math::sqrt(p.template get<real_t>("scales.sigma0")) }
      , init_flds { wave_E_ovr_B, wave_k } {}

    inline PGen() {}

    inline void InitPrtls(Domain<S, M>& domain) {
      const auto maxwellian   = arch::Maxwellian<S, M>(domain.mesh.metric,
                                                     domain.random_pool(),
                                                     temperature);
      const auto density_wave = DensityWave<S, M>(domain.mesh.metric,
                                                  delta_n_electron,
                                                  wave_k);

      arch::InjectNonUniform<S, M, decltype(maxwellian), decltype(maxwellian), decltype(density_wave)>(
        params,
        domain,
        { 1, 2 },
        { maxwellian, maxwellian },
        density_wave,
        TWO);

      // set ions as static neutralizing background
      domain.species[1].set_npart(0);
    }
  };
} // namespace user
#endif













