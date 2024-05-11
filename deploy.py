import streamlit as st
import multiprocessing

must_reload_page = False

def start_flask():
    if not hasattr(st, 'already_started_server'):
        st.already_started_server = True
        must_reload_page = True

        from flask import Flask

        app = Flask(__name__)
        
                
        @app.route('/foo')
        def serve_foo():
            return 'This page is served via Flask!'

        app.run(port=8888)

def reload_page():
    if must_reload_page:
        must_reload_page = False
        st.experimental_rerun()

if __name__=='__main__':
    flask_process = multiprocessing.Process(target=start_flask)
    reload_process = multiprocessing.Process(target=reload_page)
    flask_process.start()
    reload_process.start()


# Your normal Streamlit app goes here:
x = st.slider('Pick a number')
st.write('You picked:', x)