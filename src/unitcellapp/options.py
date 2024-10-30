from unitcellengine.analysis.homogenization import Ei, Ki, Gij, nuij
import json
from copy import copy

# Plottable items
NCUSTOM = 20


def identity(ref):
    return ref[0]


_OPTIONS = dict(
    relativeDensity=dict(
        name="Relative density",
        info="Defines the volume of material in the unit cell relative to the total volume of the unit cell itself. This property is particularly useful for not only characterizing the mass efficiency, but is also one of the dominant properties that affects unit cell performance.",
        ref=["relativeDensity"],
        calc=identity,
    ),
    E1=dict(
        name="Rel. elastic modulus in direction X",
        info="Defines the effective elastic modulus of the unit cell in the 'X' loading direction. This quantity is normalized according to the elastic modulus of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the modulus of elasticity of aluminum (68 GPa).",
        ref=["homogenizedStiffness"],
        calc=lambda ref: Ei(ref[0], [1, 0, 0]),
    ),
    E2=dict(
        name="Rel. elastic modulus in direction Y",
        info="Defines the effective elastic modulus of the unit cell in the 'Y' loading direction. This quantity is normalized according to the elastic modulus of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the modulus of elasticity of aluminum (68 GPa).",
        ref=["homogenizedStiffness"],
        calc=lambda ref: Ei(ref[0], [0, 1, 0]),
    ),
    E3=dict(
        name="Rel. elastic modulus in direction Z",
        info="Defines the effective elastic modulus of the unit cell in the 'Z' loading direction. This quantity is normalized according to the elastic modulus of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the modulus of elasticity of aluminum (68 GPa).",
        ref=["homogenizedStiffness"],
        calc=lambda ref: Ei(ref[0], [0, 0, 1]),
    ),
    Emin=dict(
        name="Min rel. elastic modulus",
        info="Defines the effective elastic modulus of the unit cell in the 'softest' possible loading direction. This quantity is normalized according to the elastic modulus of the base material. For example, to determine the absolute effective modulus of an aluminum lattice, you would simply multiple this value by the modulus of elasticity of aluminum (68 GPa).",
        ref=["Emin"],
        calc=identity,
    ),
    Emax=dict(
        name="Max rel. elastic modulus",
        info="Defines the effective elastic stiffness of the unit cell in the 'stiffest' possible loading direction. This quantity is normalized according to the elastic modulus of the base material. For example, to determine the absolute effective modulus of an aluminum lattice, you would simply multiple this value by the modulus of elasticity of aluminum (68 GPa).",
        ref=["Emax"],
        calc=identity,
    ),
    G12=dict(
        name="Rel. shear modulus in direction X on face Y",
        info="Defines the effective shear modulus of the unit cell in the 'X' direction on the 'Y' face. This quantity is normalized according to the elastic modulus of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the shear modulus of aluminum (68 GPa).",
        ref=["homogenizedStiffness"],
        calc=lambda ref: Gij(ref[0], [1, 0, 0], [0, 1, 0]),
    ),
    G23=dict(
        name="Rel. shear modulus in direction Y on face Z",
        info="Defines the effective shear modulus of the unit cell in the 'Y' direction on the 'Z' face. This quantity is normalized according to the elastic modulus of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the shear modulus of aluminum (68 GPa).",
        ref=["homogenizedStiffness"],
        calc=lambda ref: Gij(ref[0], [0, 1, 0], [0, 0, 1]),
    ),
    G13=dict(
        name="Rel. shear modulus in direction X on face Z",
        info="Defines the effective shear modulus of the unit cell in the 'X' direction on the 'Z' face. This quantity is normalized according to the elastic modulus of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the shear modulus of aluminum (68 GPa).",
        ref=["homogenizedStiffness"],
        calc=lambda ref: Gij(ref[0], [1, 0, 0], [0, 0, 1]),
    ),
    Gmin=dict(
        name="Min rel. shear modulus",
        info="Defines the effective shear modulus of the unit cell in the 'softest' possible loading direction. This quantity is normalized according to the elastic modulus of the base material. For example, to determine the absolute effective shear modulus of an aluminum lattice, you would simply multiple this value by the shear modulus of aluminum (68 GPa).",
        ref=["Gmin"],
        calc=identity,
    ),
    Gmax=dict(
        name="Max rel. shear modulus",
        info="Defines the effective shear stiffness of the unit cell in the 'stiffest' possible loading direction. This quantity is normalized according to the elastic modulus of the base material. For example, to determine the absolute effective shear modulus of an aluminum lattice, you would simply multiple this value by the shear modulus of aluminum (68 GPa).",
        ref=["Gmax"],
        calc=identity,
    ),
    nu12=dict(
        name="Poisson's ratio in direction X on face Y",
        info="Defines the effective Poisson's ratio of the unit cell in the 'X' direction on the 'Y' face.",
        ref=["homogenizedStiffness"],
        calc=lambda ref: nuij(ref[0], [1, 0, 0], [0, 1, 0]),
    ),
    nu23=dict(
        name="Poisson's ratio in direction Y on face Z",
        info="Defines the effective Poisson's ratio of the unit cell in the 'Y' direction on the 'Z' face.",
        ref=["homogenizedStiffness"],
        calc=lambda ref: nuij(ref[0], [0, 1, 0], [0, 0, 1]),
    ),
    nu13=dict(
        name="Poisson's ratio in direction X on face Z",
        info="Defines the effective Poisson's ratio of the unit cell in the 'X' direction on the 'Z' face.",
        ref=["homogenizedStiffness"],
        calc=lambda ref: nuij(ref[0], [1, 0, 0], [0, 0, 1]),
    ),
    numin=dict(
        name="Min rel. Poisson's ratio",
        info="Defines the effective Poisson's ratio of the unit cell in the 'softest' possible loading direction.",
        ref=["numin"],
        calc=identity,
    ),
    numax=dict(
        name="Max rel. Poisson's ratio",
        info="Defines the effective Poisson's ratio of the unit cell in the 'stiffest' possible loading direction.",
        ref=["numax"],
        calc=identity,
    ),
    vonMisesWorst11=dict(
        name="Max stress amplification: x",
        info="Defines the amplification factor of the external loading to the local geometry when the unit cell is loaded in the x direction. For example, is a lattice is macroscopically loaded with a stress of 100 MPa and the amplification factor is 10, then the local deformation will result in 1000 MPa at the worst case location within the unit cell.",
        ref=["vonMisesWorst11"],
        calc=identity,
    ),
    vonMisesWorst22=dict(
        name="Max stress amplification: y",
        info="Defines the amplification factor of the external loading to the local geometry when the unit cell is loaded in the y direction. For example, is a lattice is macroscopically loaded with a stress of 100 MPa and the amplification factor is 10, then the local deformation will result in 1000 MPa at the worst case location within the unit cell.",
        ref=["vonMisesWorst22"],
        calc=identity,
    ),
    vonMisesWorst33=dict(
        name="Max stress amplification: z",
        info="Defines the amplification factor of the external loading to the local geometry when the unit cell is loaded in the z direction. For example, is a lattice is macroscopically loaded with a stress of 100 MPa and the amplification factor is 10, then the local deformation will result in 1000 MPa at the worst case location within the unit cell.",
        ref=["vonMisesWorst33"],
        calc=identity,
    ),
    vonMisesWorst12=dict(
        name="Max stress amplification: xy",
        info="Defines the amplification factor of the external loading to the local geometry when the unit cell is loaded in x/y shear. For example, is a lattice is macroscopically loaded with a shear stress of 100 MPa and the amplification factor is 10, then the local deformation will result in 1000 MPa at the worst case location within the unit cell.",
        ref=["vonMisesWorst12"],
        calc=identity,
    ),
    vonMisesWorst23=dict(
        name="Max stress amplification: yz",
        info="Defines the amplification factor of the external loading to the local geometry when the unit cell is loaded in y/z shear. For example, is a lattice is macroscopically loaded with a shear stress of 100 MPa and the amplification factor is 10, then the local deformation will result in 1000 MPa at the worst case location within the unit cell.",
        ref=["vonMisesWorst23"],
        calc=identity,
    ),
    vonMisesWorst13=dict(
        name="Max stress amplification: xz",
        info="Defines the amplification factor of the external loading to the local geometry when the unit cell is loaded in x/z shear. For example, is a lattice is macroscopically loaded with a shear stress of 100 MPa and the amplification factor is 10, then the local deformation will result in 1000 MPa at the worst case location within the unit cell.",
        ref=["vonMisesWorst13"],
        calc=identity,
    ),
    anisotropyIndex=dict(
        name="Anisotropy index",
        info="Defines the anisotropy index (AI) for the unit cell elastic response, where AI = 0 is isotropic and increasing AI increases in anisotropy. For more details, see Ranganathan, S. I., and Ostoja-Starzewski, M. (2008). Universal elastic anisotropy index. Physical Review Letters, 101(5), 3–6. (https://doi.org/10.1103/PhysRevLett.101.055504).",
        ref=["anisotropyIndex"],
        calc=identity,
    ),
    k1=dict(
        name="Rel. thermal conductance in direction X",
        info="Defines the effective conductance of the unit cell in the 'X' direction. This quantity is normalized according to the isotropic conductance of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the conductance of aluminum (152 W/mK).",
        ref=["homogenizedConductance"],
        calc=lambda ref: ref[0][0, 0],
    ),
    k2=dict(
        name="Rel. thermal conductance in direction Y",
        info="Defines the effective conductance of the unit cell in the 'Y' direction. This quantity is normalized according to the isotropic conductance of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the conductance of aluminum (152 W/mK).",
        ref=["homogenizedConductance"],
        calc=lambda ref: ref[0][1, 1],
    ),
    k3=dict(
        name="Rel. thermal conductance in direction Z",
        info="Defines the effective conductance of the unit cell in the 'Z' direction. This quantity is normalized according to the isotropic conductance of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the conductance of aluminum (152 W/mK).",
        ref=["homogenizedConductance"],
        calc=lambda ref: ref[0][2, 2],
    ),
    kmax=dict(
        name="Max rel. thermal conductance",
        info="Defines the maximum effective conductance of the unit cell. This quantity is normalized according to the isotropic conductance of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the conductance of aluminum (152 W/mK).",
        ref=["homogenizedConductance"],
        calc=lambda ref: ref[0][[0, 1, 2], [0, 1, 2]].max(),
    ),
    kmin=dict(
        name="Min rel. thermal conductance",
        info="Defines the minimum effective conductance of the unit cell. This quantity is normalized according to the isotropic conductance of the base material. For example, to specialize this parameter for an aluminum lattice, you would simply multiple this value by the conductance of aluminum (152 W/mK).",
        ref=["homogenizedConductance"],
        calc=lambda ref: ref[0][[0, 1, 2], [0, 1, 2]].min(),
    ),
    relativeSurfaceArea=dict(
        name="Relative surface area",
        info="Defines the interior surface area of the unit cell relative to the total surface area of the unit cell bounding box (2[LW+WH+HL]).",
        ref=["relativeSurfaceArea"],
        calc=identity,
    ),
    xyAR=dict(
        name="X/Y aspect ratio",
        info="Ratio of unit length (X direction) to width (Y direction)",
        ref=["length", "width"],
        calc=lambda ref: ref[0] / ref[1],
    ),
    yzAR=dict(
        name="Y/Z aspect ratio",
        info="Ratio of unit width (Y direction) to height (Z direction)",
        ref=["width", "height"],
        calc=lambda ref: ref[0] / ref[1],
    ),
    xzAR=dict(
        name="X/Z aspect ratio",
        info="Ratio of unit length (X direction) to height (Z direction)",
        ref=["length", "height"],
        calc=lambda ref: ref[0] / ref[1],
    ),
    length=dict(
        name="Relative length",
        info="Normalized length of the unit cell (X direction). This quantity is only relevant when compared against other normalized geometric properties, such as 'Relative width' and/or 'Relative height' of a unit cell. For example, if the desired unit cell with have a width of 3 mm and the relative length, width, and height, are 2, 1, and 3, respectively. Then, the actual unit cell geometry is 6 mm x 3 mm x 9 mm",
        ref=["length"],
        calc=identity,
    ),
    width=dict(
        name="Relative width",
        info="Normalized width of the unit cell (Y direction). This quantity is only relevant when compared against other normalized geometric properties, such as 'Relative length' and/or 'Relative height' of a unit cell. For example, if the desired unit cell with have a width of 3 mm and the relative length, width, and height, are 2, 1, and 3, respectively. Then, the actual unit cell geometry is 6 mm x 3 mm x 9 mm",
        ref=["width"],
        calc=identity,
    ),
    height=dict(
        name="Relative height",
        info="Normalized height of the unit cell (Z direction). This quantity is only relevant when compared against other normalized geometric properties, such as 'Relative height' and/or 'Relative height' of a unit cell. For example, if the desired unit cell with have a width of 3 mm and the relative length, width, and height, are 2, 1, and 3, respectively. Then, the actual unit cell geometry is 6 mm x 3 mm x 9 mm",
        ref=["height"],
        calc=identity,
    ),
    thickness=dict(
        name="Relative thickness",
        info="Normalized thickness of the unit cell. This quantity is only relevant when compared against other normalized geometric properties, such as 'Relative width' and/or 'Relative width' and/or 'Relative height' of a unit cell. For example, if the relative thickness is 0.1 and the relative length is 2, then the unit cell thickness is 5% that of the unit cell length.",
        ref=["thickness"],
        calc=identity,
    ),
    radius=dict(
        name="Relative fillet radius",
        info="Normalized fillet radius of unit cell junction points with respect to the 'thickness'. This quantity is only relevant when compared against other normalized geometric properties, such as 'Relative length' and/or 'Relative width' and/or 'Relative height' of a unit cell. For example, if the relative radius is 0.1, then the unit cell fillet radius is 10% that of the unit cell ligament/plate thickness.",
        ref=["thickness"],
        calc=identity,
    ),
)
_CUSTOM_QOI_INFO = (
    "User defined equation defining a new QOI based on " "pre-exisiting QOI."
)


def _options(custom):
    """QOI options including custom definitions

    Arguments
    ---------
    custom: dict or serialized json string
        This consists of a (potentially serialized) dictionary with
        integer keys. For each integer key, there is a dictionary that contains
        a name and info key for a QOI.

    Returns
    -------
    options: dictionary of QOI options
    """

    # Copy the static options
    options = copy(_OPTIONS)

    # Deserialize the custom JSON data if necessary
    if not isinstance(custom, dict):
        custom = json.loads(custom)

    # Add in the custom values
    options.update(
        {
            f"custom{ind}": dict(
                name=custom[str(ind)]["name"],
                info=custom[str(ind)]["info"],
                ref=[],
                calc=lambda x: 1,
            )
            for ind in range(1, NCUSTOM + 1)
        }
    )

    return options


_DEFAULT_CUSTOM = {
    str(ind): dict(name=f"Custom QOI {ind}", info=_CUSTOM_QOI_INFO)
    for ind in range(1, NCUSTOM + 1)
}
_DEFAULT_OPTIONS = _options(_DEFAULT_CUSTOM)

# _OPTIONS.update({f"custom{ind}": dict(name=f"Custom QOI {ind}",
#                                       info=_CUSTOM_QOI_INFO,
#                                       ref=[],
#                                       calc=lambda x: 1,)
#                             for ind in range(1, NCUSTOM+1)})

OPTIONS = lambda options: [
    {"label": l["name"], "value": v, "title": l["info"] + f" (variable name: {v})"}
    for v, l in options.items()
]
OPTIONS_NORMALIZE = lambda options: [
    {
        "label": "None",
        "value": 0,
        "title": "Do not normalize the quantities of interest",
    }
] + OPTIONS(options)
