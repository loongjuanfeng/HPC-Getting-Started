{
  description = "C++23 HPC getting-started kernels";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.gcc14Stdenv.mkDerivation {
            pname = "hpc-getting-started";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              pkg-config
            ];

            buildInputs = with pkgs; [
              cli11
              openblas
              spdlog
            ];

            cmakeFlags = [
              "-DHPC_ENABLE_CUDA=OFF"
              "-DUSE_NATIVE_ARCH=OFF"
            ];

            installPhase = ''
              runHook preInstall

              mkdir -p "$out/bin"
              cp \
                vector_addition-openmp \
                matrix_multiplication-handwritten \
                matrix_multiplication-openblas \
                stencil_2d-baseline \
                stencil_2d-openmp \
                stencil_2d-tiled \
                spmv_csr-baseline \
                spmv_csr-openmp \
                spmv_csr-balanced \
                "$out/bin/"

              runHook postInstall
            '';
          };
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cmake
              ninja
              pkg-config
              gcc14
              clang-tools
              cli11
              openblas
              spdlog
            ];

            shellHook = ''
              echo "HPC dev shell ready: use make configure && make build"
            '';
          };
        }
      );
    };
}
