from bs4 import BeautifulSoup
import sys
import subprocess
import os

from pathlib import Path

# python notebook_parser.py {path_to_notebook.ipynb} (show|all|hide) {remote_path}


if len(sys.argv) == 4:
    
    path = Path(sys.argv[1])
    mode = sys.argv[2]    
    destination = sys.argv[3]

    if not path.exists():
        print(f"The path provided ({sys.argv[1]}) does not exist")
        exit()
    
    if mode not in ('show', 'hide', 'none'):
        print(f"Unknown clean-up mode '{mode}'. Should be one of 'show', 'hide' or 'none'")
        exit()

    print("Running nbconvert...")
    temp_location = 'temp.html'
    command = f"jupyter nbconvert --to html --output {temp_location} {'--no-input' if mode == 'none' else ''} {path}"    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print('nbconvert failed')
        print(result.stdout)
        print(result.stderr)
        exit()
    else:
        print('nbconvert succeeded')

    print("Parsing generated HTML...")
    output_folder = path.parent.joinpath('html')    
    if not output_folder.exists():
        output_folder.mkdir()

    main_filename = path.parent.name

    with open(path.parent.joinpath(temp_location)) as f:
        document = BeautifulSoup("\n".join(f.readlines()), "lxml")

        #save all the styles
        styles = document.find_all('style')
        with open(output_folder.joinpath('styles.css'), 'w') as style_file:
            for s in styles:            
                style_file.writelines(s.contents)
                s.decompose()

        #add styles reference to the main document
        style_ref = document.new_tag("link", href="styles.css", rel="stylesheet")
        document.find("head").append(style_ref)
        
        #remove warning outputs
        for w in document.select('div[data-mime-type="application/vnd.jupyter.stderr"]'):
            w.decompose()

        #remove code cells to be hidden
        for c in document.select(".jp-CodeCell .jp-Cell-inputWrapper"):        
                remove = (mode == 'show')
                for comm in c.select(".c1"):  
                    if mode == 'all':
                        remove = True
                        break
                    elif (comm.contents[0][:5] == "#HIDE"):            
                        remove = True
                        break
                    elif (comm.contents[0][:5] == "#SHOW"):            
                        remove = False
                        comm.contents[0] = comm.contents[5:]
                        break

                if remove:            
                    parent = c.findParent('div')
                    c.decompose()                    
                    if len(parent.findChildren()) == 0:
                        parent.decompose()

        for d in document.select('.jp-OutputPrompt'):
            d.decompose()

        for d in document.select('.jp-InputPrompt'):
            d.decompose()

        #save cleaned document
        with open(output_folder.joinpath(main_filename + ".html"), "w") as main_file:
            main_file.writelines(str(document))

    print("Saved generated HTML")
    os.remove(path.parent.joinpath(temp_location))
    print("Uploading latest version to server...")
    upload_command = f"scp {output_folder.joinpath(main_filename + '.html')} {destination}"
    result = subprocess.run(upload_command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print('upload failed')
        print(result.stdout)
        print(result.stderr)
        exit()
    else:        
        print("DONE!")
else:
    print('Not enough arguments')
    print("".join(sys.argv))