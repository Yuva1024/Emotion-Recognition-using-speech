#!/usr/bin/env bash
mkdir -p ~/.streamlit/

echo "\
[global]\n\
developmentMode = false\n\
\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = \$PORT\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = '#667eea'\n\
backgroundColor = '#ffffff'\n\
secondaryBackgroundColor = '#f0f2f6'\n\
textColor = '#262730'\n\
" > ~/.streamlit/config.toml 