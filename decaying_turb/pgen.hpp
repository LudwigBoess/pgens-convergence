#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include <stdlib.h>
#endif // MPI_ENABLED

namespace user {
  using namespace ntt;

  // initializing guide field and curl(B) = J_ext at the initial time step
  template <Dimension D>
  struct InitFields {
    InitFields(real_t                            dB,
              std::vector<std::vector<real_t>>& wavenumbers,
              unsigned int                      seed,
              real_t                            Lx,
              real_t                            Ly,
              real_t                            Lz)
      : dB { dB } 
      , wavenumbers { wavenumbers }
      , n_modes { wavenumbers.size() }
      , seed { seed }
      , Lx { Lx }
      , Ly { Ly }
      , Lz { Lz }
      , k { "wavevector", D, n_modes }
      , A0 { "A0", n_modes }
      , phase { "phase", n_modes } {
      // initializing random generator for phases
      srand(seed);
      // initializing wavevectors
      auto k_host = Kokkos::create_mirror_view(k);
      if constexpr (D == Dim::_2D) {
        for (auto i = 0u; i < n_modes; i++) {
          k_host(0, i) = constant::TWO_PI * wavenumbers[i][0] / Lx;
          k_host(1, i) = constant::TWO_PI * wavenumbers[i][1] / Ly;
        }
      }

      // initializing amplitudes and phases
      auto A0_host         = Kokkos::create_mirror_view(A0);
      auto phase_host      = Kokkos::create_mirror_view(phase);
      for (auto i = 0u; i < n_modes; i++) {
        auto k_perp = math::sqrt(
          k_host(0, i) * k_host(0, i) + k_host(1, i) * k_host(1, i));
	      phase_host(i) = static_cast <real_t> (rand()) / static_cast <real_t> (RAND_MAX) * constant::TWO_PI;
        A0_host(i) = dB / math::sqrt((real_t)n_modes) / k_perp;
      };
      Kokkos::deep_copy(A0, A0_host);
      Kokkos::deep_copy(k, k_host);
      Kokkos::deep_copy(phase, phase_host);
    };

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      auto bx1_0 = ZERO;
      if constexpr(D==Dim::_2D){
        for (auto i = 0; i < n_modes; i++) {
          auto k_dot_r  = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1];
          bx1_0        -= TWO * k(1, i) * A0(i) * math::sin( k_dot_r + phase(i) );
        }
      }
      return bx1_0;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      auto bx2_0 = ZERO;
      if constexpr (D==Dim::_2D){
        for (auto i = 0; i < n_modes; i++) {
          auto k_dot_r  = k(0, i) * x_Ph[0] + k(1, i) * x_Ph[1];
          bx2_0        += TWO * k(0, i) * A0(i) * math::sin( k_dot_r + phase(i) );
        }
      }
      return bx2_0;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ONE;
    }
  private:
    const std::vector<std::vector<real_t>> wavenumbers;
    const std::size_t                      n_modes;
    const real_t                           dB, Lx, Ly, Lz;
    const unsigned int                     seed; 

  public:
    array_t<real_t**> k;
    array_t<real_t*>  phase; 
    array_t<real_t*>  A0;
  };

  inline auto init_pool(int seed) -> unsigned int {
    if (seed == 0) {
      unsigned int new_seed = static_cast<unsigned int>(rand());
#if defined(MPI_ENABLED)
      MPI_Bcast(&new_seed, 1, MPI_UNSIGNED, MPI_ROOT_RANK, MPI_COMM_WORLD);
#endif // MPI_ENABLED
      return new_seed;
    } else {
      return static_cast<unsigned int>(seed);
    }
  }

  template <Dimension D>
  inline auto init_wavenumbers() -> std::vector<std::vector<real_t>> {
      if constexpr (D == Dim::_2D) {
          std::vector<std::vector<real_t>> wavenumbers;
          for (int y = 1; y <= 4; y++) {
              for (int x = 1; x <= 4; x++) {
  
                  if (x == 0 && y == 0)
                      continue;
  
                  wavenumbers.push_back({
                      static_cast<real_t>(x),
                      static_cast<real_t>(y)
                  });
              }
          }
          return wavenumbers;
  
      } else if constexpr (D == Dim::_3D) {
          return {
              {  1,  0,  1 },
              {  0,  1,  1 },
              { -1,  0,  1 },
              {  0, -1,  1 },
              {  1,  0, -1 },
              {  0,  1, -1 },
              { -1,  0, -1 },
              {  0, -1, -1 }
          };
  
      } else {
          raise::Error("Invalid dimension", HERE);
          return {};
      }
  }

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t                     temperature, dB;
    const real_t                     Lx, Ly, Lz;
    const unsigned int               random_seed;
    std::vector<std::vector<real_t>> wavenumbers;
    random_number_pool_t             random_pool;

    // debugging, will delete later
    real_t total_sum           = ZERO;
    real_t total_sum_inv       = ZERO;
    real_t number_of_timesteps = ZERO;

    InitFields<D>      init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , temperature { p.template get<real_t>("setup.temperature") }
      , dB { p.template get<real_t>("setup.dB", ONE) }
      , wavenumbers { init_wavenumbers<D>() }
      , random_seed { p.template get<unsigned int>("setup.seed", 42) }
      , random_pool { init_pool(random_seed) }
      , Lx { global_domain.mesh().extent(in::x1).second -
             global_domain.mesh().extent(in::x1).first }
      , Ly { global_domain.mesh().extent(in::x2).second -
             global_domain.mesh().extent(in::x2).first }
      , Lz { global_domain.mesh().extent(in::x3).second -
             global_domain.mesh().extent(in::x3).first }
      , init_flds { dB, wavenumbers, random_seed, Lx, Ly, Lz } {};

      inline void InitPrtls(Domain<S, M>& domain) {
        arch::InjectUniformMaxwellian<S, M>(params, domain, ONE, temperature, { 1, 2 });
      };
  };
} // namespace user

#endif
