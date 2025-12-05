# Aryan's Plotting Preferences and Style Guide

## Overview
This document outlines preferred plotting styles and configuration for scientific/technical plots. Use this as a reference when creating visualizations to avoid iterative refinements.

## Reference Example
See: `cpts530/evaluations/final/src/plot_p02.jl` and the generated `p02_trajectory_comparison.png`

This example demonstrates all key preferences implemented correctly.

## Core Requirements

### 1. Typography and LaTeX Formatting
**Critical:** Always use LaTeX formatting for mathematical expressions.

```julia
using LaTeXStrings

# Titles with equations - use L"..." for LaTeX strings
title="Problem Title\n" * L"(e^t + 1)x' + xe^t - x = 0, x(0) = 3"

# Axis labels - use LaTeX even for simple expressions
xlabel=L"Time $t$"
ylabel=L"$x(t)$"

# NOT like this:
xlabel="Time t"        # ❌ Plain text
ylabel="x(t)"          # ❌ Plain text
```

### 2. Grid Configuration
**Always include both major and minor grids** with subtle, harmonious styling.

```julia
plot(
    # ... other parameters ...
    grid=true,
    minorgrid=true,
    gridcolor=soft_pink,              # Or similar soft color for major grid
    minorgridcolor=soft_lavender,     # Or similar soft color for minor grid
    gridlinewidth=1.5,
    minorgridlinewidth=0.8,
    gridalpha=0.5,
    minorgridalpha=0.3,
    gridstyle=:solid,                 # Solid lines, not dots or dashes
)
```

### 3. Line Styling
Use appropriate line weights to create visual hierarchy:

```julia
# Analytical/exact solutions - thicker, solid
plot!(t, x_exact,
    label="Analytical Solution",
    linewidth=4,                      # Thick for emphasis
    linestyle=:solid,
    color=primary_color
)

# Numerical/approximate solutions - slightly thinner, dashed
plot!(t, x_numerical,
    label="RK4 Numerical Solution",
    linewidth=3,                      # Slightly thinner
    linestyle=:dash,                  # Dashed to distinguish
    color=secondary_color
)
```

### 4. Color Scheme
While specific colors may vary by project, follow these principles:

- **Use contrasting colors** that are distinguishable
- **Prefer soft, professional colors** over bright/garish ones
- **Maintain consistency** within a project
- **Example palette:**
  ```julia
  primary = "#eb6f92"      # Rose/pink for analytical
  secondary = "#31748f"    # Teal/blue for numerical
  background = "#FFFFFF"   # Pure white
  text = "#000000"         # Black for text/ticks
  ```

### 5. Background and Foreground
**Always use clean, high-contrast backgrounds:**

```julia
plot(
    background_color="#FFFFFF",       # Pure white, not off-white
    foreground_color="#000000",       # Black for all text elements
)
```

### 6. Font Styling
Apply consistent font colors to all text elements:

```julia
plot(
    legendfontcolor=text_color,
    legendfontsize=11,
    titlefontsize=14,
    titlefontcolor=text_color,
    labelfontsize=13,
    guidefontcolor=text_color,        # Axis labels
    tickfontcolor=text_color,
    tickfontsize=11,
)
```

### 7. Legend Placement
Position legend appropriately based on data:

```julia
legend=:topleft      # Or :topright, :bottomleft, :bottomright
                     # Choose position that doesn't obscure data
```

### 8. Plot Dimensions and Quality

```julia
plot(
    size=(800, 600),     # Standard aspect ratio
    dpi=300,             # High quality for publications
    margin=5Plots.mm,    # Adequate spacing around plot
    framestyle=:box,     # Complete box frame
)
```

### 9. Annotations
When adding text annotations:

```julia
annotate!(
    x_pos, y_pos,
    text("Step size h = -0.01\n200 steps", 
         annotation_color, 
         10,           # Font size
         :left)        # Alignment
)
```

## Complete Template

```julia
using Plots
using LaTeXStrings

# Define color scheme
primary = "#eb6f92"
secondary = "#31748f"
bg_color = "#FFFFFF"
text_color = "#000000"
grid_major = "#E8C4D4"    # Soft pink
grid_minor = "#D4D4E8"    # Soft lavender

# Create plot
plot(
    x_data, y_analytical,
    label="Analytical Solution",
    linewidth=4,
    linestyle=:solid,
    color=primary,
    legend=:topleft,
    legendfontcolor=text_color,
    legendfontsize=11,
    xlabel=L"Time $t$",
    ylabel=L"$x(t)$",
    title="Problem Title\n" * L"Mathematical Expression",
    titlefontsize=14,
    titlefontcolor=text_color,
    labelfontsize=13,
    guidefontcolor=text_color,
    tickfontcolor=text_color,
    tickfontsize=11,
    background_color=bg_color,
    foreground_color=text_color,
    grid=true,
    minorgrid=true,
    gridcolor=grid_major,
    minorgridcolor=grid_minor,
    gridlinewidth=1.5,
    minorgridlinewidth=0.8,
    gridalpha=0.5,
    minorgridalpha=0.3,
    gridstyle=:solid,
    framestyle=:box,
    size=(800, 600),
    dpi=300,
    margin=5Plots.mm
)

# Add additional data
plot!(
    x_data, y_numerical,
    label="Numerical Solution",
    linewidth=3,
    linestyle=:dash,
    color=secondary
)

# Optional: Add annotation
annotate!(
    x_pos, y_pos,
    text("Additional info", primary, 10, :left)
)

# Save
savefig("output.png")
```

## Common Mistakes to Avoid

1. ❌ **Using plain text instead of LaTeX**
   ```julia
   xlabel="x(t)"  # NO
   xlabel=L"$x(t)$"  # YES
   ```

2. ❌ **Omitting minor grid**
   ```julia
   grid=true  # Incomplete - missing minorgrid
   ```

3. ❌ **Using default line widths** (too thin)
   ```julia
   linewidth=1  # NO - too thin
   linewidth=3  # YES - or 4 for primary lines
   ```

4. ❌ **Forgetting to set text colors consistently**
   ```julia
   # Must set: legendfontcolor, titlefontcolor, guidefontcolor, tickfontcolor
   ```

5. ❌ **Using off-white backgrounds** instead of pure white
   ```julia
   background_color="#F5F5F5"  # NO - grayish tint
   background_color="#FFFFFF"  # YES - pure white
   ```

6. ❌ **Not using proper dpi for publication quality**
   ```julia
   dpi=100  # NO - low quality
   dpi=300  # YES - publication quality
   ```

## Testing Your Plot

Before finalizing, verify:

- [ ] All mathematical expressions use LaTeX (L"..." strings)
- [ ] Both major and minor grids are visible and styled
- [ ] Line widths create clear visual hierarchy (4 for primary, 3 for secondary)
- [ ] Background is pure white (#FFFFFF)
- [ ] All text elements (legend, title, labels, ticks) use consistent colors
- [ ] DPI is set to 300
- [ ] Legend doesn't obscure important data
- [ ] Grid is subtle but visible (alpha ~0.3-0.5)

## Notes for AI Assistants

When Aryan asks for a plot:

1. **Start with the complete template above** - don't create minimalist plots
2. **Ask about color preferences** if not specified, but apply all other settings
3. **Always include LaTeX formatting** - this is non-negotiable
4. **Include both grids** by default - major and minor
5. **Use thick lines** (3-4 width) - not the default thin lines
6. **Set high DPI** (300) for publication quality

The goal is to produce **publication-ready plots on the first attempt**, not plots that require multiple iterations of "make the lines thicker", "add LaTeX formatting", "add minor grid", etc.
