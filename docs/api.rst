API Reference
============

This page provides detailed API documentation for the main FIRM3D modules and classes.

Magnetic Field Classes
---------------------

.. automodule:: firm3d.field.boozermagneticfield
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: firm3d.field.coordinates
   :members:
   :undoc-members:
   :show-inheritance:

Trajectory Integration
---------------------

.. automodule:: firm3d.field.trajectory_helpers
   :members:
   :undoc-members:
   :show-inheritance:

Stopping Criteria
----------------

# Note: This module may not exist yet in the current codebase
# .. automodule:: firm3d.field.stopping_criteria
#    :members:
#    :undoc-members:
#    :show-inheritance:

Shear Alfvén Wave Classes
------------------------

# Note: This module may not exist yet in the current codebase
# .. automodule:: firm3d.field.shear_alfven_waves
#    :members:
#    :undoc-members:
#    :show-inheritance:

Utility Functions
----------------

.. automodule:: firm3d.util.constants
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: firm3d.util.functions
   :members:
   :undoc-members:
   :show-inheritance:

Plotting Utilities
-----------------

.. automodule:: firm3d.plotting.plotting_helpers
   :members:
   :undoc-members:
   :show-inheritance:

Core Types and Utilities
-----------------------

.. automodule:: firm3d._core.types
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: firm3d._core.util
   :members:
   :undoc-members:
   :show-inheritance:

SAW (Shear Alfvén Wave) Module
-----------------------------

.. automodule:: firm3d.saw.ae3d
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: firm3d.saw.stellgap
   :members:
   :undoc-members:
   :show-inheritance:

Class Hierarchy
--------------

.. inheritance-diagram:: firm3d.field.boozermagneticfield
   :parts: 1

.. inheritance-diagram:: firm3d.field.trajectory_helpers
   :parts: 1

# .. inheritance-diagram:: firm3d.field.stopping_criteria
#    :parts: 1

# Function Index
# -------------
#
# .. autosummary::
#    :toctree: _autosummary
#    :template: function.rst
#    :recursive:
#
#    firm3d.field.trajectory_helpers.trace_particles_boozer
#    firm3d.field.trajectory_helpers.trace_particles_boozer_perturbed
#    firm3d.field.boozermagneticfield.BoozerAnalytic
#    firm3d.field.boozermagneticfield.BoozerRadialInterpolant
#    firm3d.field.boozermagneticfield.InterpolatedBoozerField
#
# Class Index
# ----------
#
# .. autosummary::
#    :toctree: _autosummary
#    :template: class.rst
#    :recursive:
#
#    firm3d.field.boozermagneticfield.BoozerMagneticField
#    firm3d.field.boozermagneticfield.BoozerAnalytic
#    firm3d.field.boozermagneticfield.BoozerRadialInterpolant
#    firm3d.field.boozermagneticfield.InterpolatedBoozerField
#    # firm3d.field.shear_alfven_waves.ShearAlfvenHarmonic
#    # firm3d.field.shear_alfven_waves.ShearAlfvenWavesSuperposition
#    # firm3d.field.stopping_criteria.StoppingCriterion
#    # firm3d.field.stopping_criteria.MaxToroidalFluxStoppingCriterion
#    # firm3d.field.stopping_criteria.MinToroidalFluxStoppingCriterion
#    # firm3d.field.stopping_criteria.ZetaStoppingCriterion
#    # firm3d.field.stopping_criteria.VparStoppingCriterion
#    # firm3d.field.stopping_criteria.ToroidalTransitStoppingCriterion
#    # firm3d.field.stopping_criteria.IterationStoppingCriterion
#    # firm3d.field.stopping_criteria.StepSizeStoppingCriterion
#
# Module Index
# -----------
#
# .. autosummary::
#    :toctree: _autosummary
#    :template: module.rst
#    :recursive:
#
#    firm3d.field
#    firm3d.field.boozermagneticfield
#    firm3d.field.coordinates
#    firm3d.field.trajectory_helpers
#    # firm3d.field.stopping_criteria
#    # firm3d.field.shear_alfven_waves
#    firm3d.util
#    firm3d.util.constants
#    firm3d.util.functions
#    firm3d.plotting
#    firm3d.plotting.plotting_helpers
#    firm3d._core
#    firm3d._core.types
#    firm3d._core.util
#    firm3d.saw
#    firm3d.saw.ae3d
#    firm3d.saw.stellgap
