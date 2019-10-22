#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author Tim Hladnik

from Translation_Stimulus_Spherical_Arena import *

frametime = .05
dur = 5.

stim = Stimulus()


# Create translation stimuli for different spatial frequencies and velocities
for sf in np.linspace(1./180, 5./180, 5):
    # Create a pattern
    pattern = Pattern.Bars(sf=sf)

    # Use multiple test velocities
    for v in np.linspace(1., 45., 4):

        # whole_field stimulus
        background = createTranslationStimulus(stim.verts,
                                               pattern=pattern, duration=dur, v=.0, frametime=frametime)
        # Forward translation movement stimulus
        pos_transl = createTranslationStimulus(stim.verts,
                                               pattern=pattern, duration=dur, v=v, frametime=frametime)
        # Backwards translation movement stimulus
        neg_transl = createTranslationStimulus(stim.verts,
                                               pattern=pattern, duration=dur, v=-v, frametime=frametime)

        # Apply masks and add phase to stimulus
        phase = applyMasks(stim.verts, background,
                           ['transl_stripe_symm', np.pi / 4, np.pi / 4, pos_transl],
                           ['transl_stripe_symm', -np.pi / 4, np.pi / 4, neg_transl],
                           )
        stim.addPhase(phase)

# Display in separate window
stim.display(frametime)

# Save to file
#stim.saveAs('example01')