#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bmag) : bmag { bmag } {}

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return bmag;
    }

  private:
    const real_t bmag;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t temperature, drift_vel, density;
    const in     drift_dir;
    const real_t width, y1, y2;

    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temperature { p.template get<real_t>("setup.temperature", 0.01) }
      , drift_vel { p.template get<real_t>("setup.sonic_mach", 1.0) *
                    math::sqrt((5 / 3) * temperature) }
      , density { p.template get<real_t>("setup.density", 1.0) }
      , drift_dir { static_cast<in>(p.template get<int>("setup.drift_dir", 0)) }
      , width { p.template get<real_t>("setup.width", 0.01) }
      , y1 { p.template get<real_t>("setup.y1", -0.25) }
      , y2 { p.template get<real_t>("setup.y2", 0.25) }
      , init_flds { drift_vel * density *
                    p.template get<real_t>("setup.inv_alfven_mach", 0.0) /
                    math::sqrt(p.template get<real_t>("scales.sigma0")) } {}

    inline void InitPrtls(Domain<S, M>& domain) {
        
      /*
        * Plasma setup as 3 individual slabs

        ______________________________________
        | .....................................|
        | ............ (+v0) ..................|   density = 1
        |--------------------------------------|
        |::::::::::::::::::::::::::::::::::::::|
        |::::::::::::::::::::::::::::::::::::::|
        |:::::::::::::: (-v0) :::::::::::::::::|   density = 2
        |::::::::::::::::::::::::::::::::::::::|
        |--------------------------------------|
        | .....................................|
        | ............ (+v0) ..................|   density = 1
        |______________________________________|

      */


      // define box to inject into
      boundaries_t<real_t> lower_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 1) {
          lower_box.push_back({ domain.mesh.extent(in::x2).first, y1});
        } else {
          lower_box.push_back(Range::All);
        }
      }

      // define box to inject into
      boundaries_t<real_t> middle_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 1) {
          middle_box.push_back({ y1, y2 });
        } else {
          middle_box.push_back(Range::All);
        }
      }

      // define box to inject into
      boundaries_t<real_t> upper_box;
      // loop over all dimension
      for (auto d = 0u; d < M::Dim; ++d) {
        if (d == 1) {
          upper_box.push_back({ y2, domain.mesh.extent(in::x2).second });
        } else {
          upper_box.push_back(Range::All);
        }
      }


      // energy distribution of the particles in the low density regions
      const auto low_density_dist  = arch::Maxwellian<S, M>(domain.mesh.metric,
                                                      domain.random_pool,
                                                      TWO * temperature,
                                                      drift_vel,
                                                      in::x1);

      const auto low_density_injector = arch::UniformInjector<S, M, arch::Maxwellian>(low_density_dist,
                                                                        { 1, 2 });


      // energy distribution of the particles in the high density regions
      const auto high_density_dist  = arch::Maxwellian<S, M>(domain.mesh.metric,
                                                      domain.random_pool,
                                                      temperature,
                                                      -drift_vel,
                                                      in::x1);

      const auto high_density_injector = arch::UniformInjector<S, M, arch::Maxwellian>(high_density_dist,
                                                                        { 1, 2 });


      // inject uniformly within the defined boxes

      // lower box
      arch::InjectUniform<S, M, decltype(low_density_injector)>(params,
                                                    domain,
                                                    low_density_injector,
                                                    density,
                                                    false,
                                                    lower_box);

      // middle box
      arch::InjectUniform<S, M, decltype(high_density_injector)>(params,
                                                    domain,
                                                    high_density_injector,
                                                    TWO * density,
                                                    false,
                                                    middle_box);

      // upper box
      arch::InjectUniform<S, M, decltype(low_density_injector)>(params,
                                                    domain,
                                                    low_density_injector,
                                                    density,
                                                    false,
                                                    upper_box);

          
    }
  };

} // namespace user

#endif
