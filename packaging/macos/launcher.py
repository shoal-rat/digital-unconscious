"""The executable inside Digital Unconscious.app.

Opened from Finder or the Dock it opens the shore; given dun's arguments (as the login
item does, or a `dun` link in a terminal) it does what `dun` does.
"""

import sys

from unconscious.env import adopt_login_env

adopt_login_env()  # PATH, proxies and keys from the login shell: the Dock passes none of them

from unconscious.cli import main  # noqa: E402  (after the environment is in place)

arguments = [arg for arg in sys.argv[1:] if not arg.startswith("-psn_")]  # Finder's process serial number
sys.exit(main(arguments or ["app"]))
