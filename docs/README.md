# FIRM3D Documentation

This directory contains the documentation for FIRM3D, built using Sphinx and hosted on ReadTheDocs.

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Local Build

To build the documentation locally:

```bash
make html
```

The built documentation will be available in `_build/html/`.

### Live Preview

To start a live server for previewing changes:

```bash
make livehtml
```

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `installation.rst` - Installation instructions
- `magnetic_fields.rst` - Magnetic field classes documentation
- `shear_alfven_waves.rst` - Shear Alfvén wave documentation
- `stellgap.rst` - Proposed STELLGAP continuum formulation and interfacen
- `guiding_center.rst` - Guiding center integration
- `stopping_criteria.rst` - Stopping criteria documentation
- `trajectory_saving.rst` - Trajectory saving options
- `magnetic_axis.rst` - Magnetic axis handling
- `solvers.rst` - Solver options and methods
- `api.rst` - API reference
- `examples.rst` - Examples and tutorials
- `contributing.rst` - Contributing guidelines

## Customization

- `_static/custom.css` - Custom CSS styles
- `_templates/layout.html` - Custom layout template
- `requirements.txt` - Documentation dependencies

## ReadTheDocs Integration

The documentation is automatically built and deployed to ReadTheDocs when changes are pushed to the main branch.

## Contributing to Documentation

1. Follow the existing style and structure
2. Use reStructuredText or Markdown format
3. Include code examples where appropriate
4. Update the table of contents in `index.rst`
5. Test the build locally before submitting changes
