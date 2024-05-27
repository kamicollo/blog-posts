from bs4 import BeautifulSoup
import argparse
import pathlib

parser = argparse.ArgumentParser(
    prog="Quarto and HTML parser",
    description="Parses HTML output produced by Quarto and makes it ready to be used in a Wordpress blog",
    epilog="",
)


parser.add_argument("filename")  # positional argument
parser.add_argument("-o", "--output-dir")  # option that takes a value
parser.add_argument("-v", "--verbose", action="store_true")  # on
args = parser.parse_args()

input_file = pathlib.Path(args.filename)
output_folder = pathlib.Path(args.output_dir)

if not input_file.exists():
    raise ValueError(f"Input file {args.filename} does not exist")

if output_folder.exists() and not output_folder.is_dir():
    raise ValueError(f"Output director {args.output_dir} is not a directory")
elif not output_folder.exists():
    output_folder.mkdir(parents=True)


with open(input_file) as f:
    document = BeautifulSoup("\n".join(f.readlines()), "lxml")

    # save all the styles
    styles = document.find_all("style")
    with open(output_folder.joinpath("quarto_styles.css"), "w") as style_file:
        for s in styles:
            style_file.writelines(s.contents)
            s.decompose()

    # delete the title
    [h.decompose() for h in document.find_all("h1")]

    # save cleaned document
    with open(output_folder.joinpath(input_file.name), "w") as main_file:
        main_file.writelines(str(document))
