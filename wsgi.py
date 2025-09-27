"""
WSGI config for Inventory Optimization Dashboard.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/stable/howto/deployment/wsgi/
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app import app as application  # noqa

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    application.run(host='0.0.0.0', port=port)
