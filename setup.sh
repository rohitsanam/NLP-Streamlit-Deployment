mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"rohit.sanam@yahoo.in\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml