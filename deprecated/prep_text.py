import tools.processing as pre
import tools.spell_correction as spell
import os

FILE = "clean2_pac.txt"

SRC = os.path.join( "data/cleaned-rap-lyrics", FILE)
DST = os.path.join("data/prepped", FILE)

text = pre.get_text( SRC )

corr_text = (spell.correct(text, "data/words_alpha.txt"))

print(corr_text[:500])

import re
corr_text = re.sub(" *linebreakhere *", "\n", corr_text)

pre.write_text( DST, corr_text)
