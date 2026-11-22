ARTIST = {
    "vangogh": "Van Gogh",
    "tyleredlin": "Tyler Edlin",
    "kilianeng": "Kilian Eng",
    "thomaskinkade": "Thomas Kinkade",
    "kellymckernan": "Kelly McKernan",
}

def paper_prompt(key):
    return "An artwork by %s" % ARTIST[key]


_VANGOGH = ("a post-impressionist oil painting, swirling brushstrokes, "
            "thick impasto, vivid blue and yellow")
_TYLEREDLIN = ("an epic fantasy environment concept art, dramatic painterly "
               "landscape, cinematic lighting")
_KILIANENG = ("a psychedelic retro-futuristic sci-fi illustration, bold flat "
              "color planes in orange teal and purple, intricate fine linework, "
              "surreal alien dreamscape, vintage 1970s science-fiction comic art")
_THOMASKINKADE = ("a luminous idyllic cottage painting, glowing golden window light "
                  "at dusk, blossoming gardens along a winding stone path, warm amber "
                  "and rose highlights, nostalgic romantic realism")
_KELLYMCKERNAN = ("an ethereal watercolor and gouache figure illustration, translucent "
                  "bleeding washes and delicate ink linework, iridescent teal magenta "
                  "and violet, flowing celestial and floral motifs, art nouveau elegance")

SPEC = {
    "prompt_reanchor": {
        "vangogh": ("$IMMA_ROOT/data/vangogh_reanchor_styledesc", _VANGOGH),
        "tyleredlin": ("$IMMA_ROOT/data/tyleredlin_reanchor_styledesc", _TYLEREDLIN),
        "kilianeng": ("$IMMA_ROOT/data/kilianeng_reanchor_styledesc", _KILIANENG),
        "thomaskinkade": ("$OUT_ROOT/data/thomaskinkade_reanchor_styledesc",
                          _THOMASKINKADE),
        "kellymckernan": ("$OUT_ROOT/data/kellymckernan_reanchor_styledesc",
                          _KELLYMCKERNAN),
    },
}

BUILDABLE = [(ch, key) for ch, spec in SPEC.items() for key, (d, _) in spec.items()
             if d.startswith("$OUT_ROOT")]


def resolve(template, imma_root, out_root):
    return template.replace("$IMMA_ROOT", imma_root).replace("$OUT_ROOT", out_root)
