using Documenter, TVPSobolev

makedocs(;
  modules = [TVPSobolev],
  sitename = "TVPSobolev.jl",
  authors = "Farhad Shahryarpoor",
  repo = "https://github.com/FarhadShahryarpoor/TVPSobolev.jl/blob/{commit}{path}#{line}",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    edit_link = "main",
    canonical = "https://FarhadShahryarpoor.github.io/TVPSobolev.jl/stable",
    repolink = "https://github.com/FarhadShahryarpoor/TVPSobolev.jl",
  ),
  pages = [
    "Home" => "index.md",
    "Methodology" => "methodology.md",
    "Reproducibility" => "reproducibility.md",
    "API Reference" => "api.md",
  ],
  checkdocs = :none,
  warnonly = [:missing_docs, :cross_references],
)

deploydocs(;
  repo = "github.com/FarhadShahryarpoor/TVPSobolev.jl",
  devbranch = "main",
)
