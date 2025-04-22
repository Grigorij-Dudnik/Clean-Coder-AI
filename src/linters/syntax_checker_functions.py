import ast
import yaml
import sass
from lxml import etree
import re
import esprima


def check_syntax(file_content, filename):
    parts = filename.split(".")
    extension = parts[-1] if len(parts) > 1 else ""
    if extension == "py":
        return parse_python(file_content)
    elif extension in ["html", "htm"]:
        return parse_html(file_content)
    elif extension == "js":
        return parse_javascript(file_content)
    elif extension in ["css", "scss"]:
        return parse_scss(file_content)
    elif extension == "vue":
        return parse_vue_basic(file_content)
    elif extension == "tsx":
        return parse_tsx(file_content)
    elif extension in ["yml", "yaml"]:
        return parse_yaml(file_content)
    else:
        return check_bracket_balance(file_content)


def parse_python(code):
    try:
        ast.parse(code)
        return "Valid syntax"
    except SyntaxError as e:
        return f"Syntax Error: {e.msg} (line {e.lineno - 1})"
    except Exception as e:
        return f"Error: {e}"


def parse_html(html_content):
    parser = etree.HTMLParser(recover=True)  # Enable recovery mode
    try:
        etree.fromstring(html_content, parser)
        significant_errors = [
            error
            for error in parser.error_log
            # Shut down some error types to be able to parse html from vue
            # if not error.message.startswith('Tag')
            # and "error parsing attribute name" not in error.message
        ]
        if not significant_errors:
            return "Valid syntax"
        else:
            for error in significant_errors:
                return f"HTML line {error.line}: {error.message}"
    except etree.XMLSyntaxError as e:
        return f"Html error occurred: {e}"


def parse_template(code):
    for tag in ["div", "p", "span", "main"]:
        function_response = check_template_tag_balance(code, f"<{tag}", f"</{tag}>")
        if function_response != "Valid syntax":
            return function_response
    return "Valid syntax"


# simple way of parsing js, to remove
def parse_javascript_basic(js_content):
    script_part_response = check_bracket_balance(js_content)
    if script_part_response != "Valid syntax":
        return script_part_response
    return "Valid syntax"


def check_template_tag_balance(code, open_tag, close_tag):
    opened_tags_count = 0
    open_tag_len = len(open_tag)
    close_tag_len = len(close_tag)
    
    i = 0
    while i < len(code):
        # check for open tag plus '>' or space after
        if code[i : i + open_tag_len] == open_tag and code[i + open_tag_len] in [" ", ">", "\n"]:
            # Look ahead for self-closing tag
            look_ahead = code[i:i + 30]  # Look ahead reasonable number of characters
            closing_pos = look_ahead.find("/>")
            
            if closing_pos != -1:
                # Self-closing tag found - skip to end of tag
                i += closing_pos + 2
            else:
                # Regular opening tag
                opened_tags_count += 1
                i += open_tag_len
        elif code[i : i + close_tag_len] == close_tag:
            opened_tags_count -= 1
            i += close_tag_len
            if opened_tags_count < 0:
                return f"Invalid syntax, mismatch of {open_tag} and {close_tag}"
        else:
            i += 1
    
    if opened_tags_count == 0:
        return "Valid syntax"
    else:
        return f"Invalid syntax, mismatch of {open_tag} and {close_tag}"


def bracket_balance(code, beginnig_bracket="{", end_bracket="}"):
    opened_brackets_count = 0

    for char in code:
        if char == beginnig_bracket:
            opened_brackets_count += 1
        elif char == end_bracket:
            opened_brackets_count -= 1
            if opened_brackets_count < 0:
                return f"Invalid syntax, mismatch of {beginnig_bracket} and {end_bracket}"

    if opened_brackets_count == 0:
        return "Valid syntax"
    else:
        return f"Invalid syntax, mismatch of {beginnig_bracket} and {end_bracket}"


def check_bracket_balance(code):
    bracket_response = bracket_balance(code, beginnig_bracket="(", end_bracket=")")
    if bracket_response != "Valid syntax":
        return bracket_response
    bracket_response = bracket_balance(code, beginnig_bracket="[", end_bracket="]")
    if bracket_response != "Valid syntax":
        return bracket_response
    bracket_response = bracket_balance(code, beginnig_bracket="{", end_bracket="}")
    if bracket_response != "Valid syntax":
        return bracket_response
    return "Valid syntax"


def parse_scss(scss_code):
    # removing import statements as they cousing error, because function has no access to filesystem
    scss_code = re.sub(r'@import\s+[\'"].*?[\'"];', "", scss_code)
    try:
        sass.compile(string=scss_code)
        return "Valid syntax"
    except sass.CompileError as e:
        return f"CSS/SCSS syntax error: {e}"


# That function does not guarantee finding all the syntax errors in template and script part; but mostly works
def parse_vue_basic(content):
    start_tag_template = re.search(r"<template>", content).end()
    end_tag_template = content.rindex("</template>")
    template = content[start_tag_template:end_tag_template]
    template_part_response = parse_template(template)
    if template_part_response != "Valid syntax":
        return template_part_response

    try:
        script = re.search(r"<script[^>]*>(.*?)</script>", content, re.DOTALL).group(1)
    except AttributeError:
        return "Script part has no valid open/closing tags."
    script_part_response = check_bracket_balance(script)
    if script_part_response != "Valid syntax":
        return script_part_response

    style_match = re.search(r"<style[^>]*>(.*?)</style>", content, re.DOTALL)
    if style_match:
        css = style_match.group(1)
        if css:  # if for the case of empty css block
            style_part_response = parse_scss(style_match.group(1))
            if style_part_response != "Valid syntax":
                return style_part_response

    return "Valid syntax"


def parse_javascript(js_content):
    try:
        esprima.parseModule(js_content)
        return "Valid syntax"
    except esprima.Error as e:
        print(f"Esprima syntax error: {e}")
        return f"JavaScript syntax error: {e}"


# Function under development
def lint_vue_code(code_string):
    import subprocess
    import os

    eslint_config_path = ".eslintrc.js"
    temp_file_path = "dzik.vue"
    # Create a temporary file
    with open(temp_file_path, "w", encoding="utf-8") as file:
        file.write(code_string)
    try:
        # Run ESLint on the temporary file
        result = subprocess.run(
            ["D:\\NodeJS\\npx.cmd", "eslint", "--config", eslint_config_path, temp_file_path, "--fix"],
            check=True,
            text=True,
            capture_output=True,
        )
        print("Linting successful:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during linting:", e.stderr)
    finally:
        # Clean up by deleting the temporary file
        os.remove(temp_file_path)


def parse_tsx(tsx_code):
    template_response = parse_template(tsx_code)
    if template_response != "Valid syntax":
        return template_response
    bracket_balance_response = check_bracket_balance(tsx_code)
    if bracket_balance_response != "Valid syntax":
        return bracket_balance_response
    return "Valid syntax"


def parse_yaml(yaml_string):
    try:
        yaml.safe_load(yaml_string)
        return "Valid syntax"
    except yaml.YAMLError as e:
        return f"YAML error: {e}"


if __name__ == "__main__":
    code = """
'use client';

import { FC, useState, useCallback, useEffect } from 'react';
import Player from './components/Player';
import Enemies from './components/Enemies';
import Projectiles from './components/Projectiles';

const GamePage: FC = () => {
  const [projectiles, setProjectiles] = useState<{ x: number; y: number }[]>([]);

  const handleShoot = useCallback((x: number) => {
    setProjectiles(prev => [...prev, { x, y: 0 }]);
  }, []);

  const handleHit = useCallback((projectileIndex: number) => {
    setProjectiles(prev => prev.filter((_, index) => index !== projectileIndex));
  }, []);

  useEffect(() => {
    const gameLoop = setInterval(() => {
      setProjectiles(prev => 
        prev
          .map(p => ({ ...p, y: p.y + 5 }))
          .filter(p => p.y < 600)
      );
    }, 16);

    return () => clearInterval(gameLoop);
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center bg-white dark:bg-[#0a0a0a]">
      <header className="w-full text-center py-8">
        <h1 className="text-4xl font-bold text-foreground">Space Invaders</h1>
      </header>

      <main className="flex-1 w-full max-w-4xl p-4">
        <div className="w-full h-[600px] bg-gray-900 rounded-lg relative">
          <Enemies projectiles={projectiles} onHit={handleHit} />
          <Player onShoot={handleShoot} />
          <Projectiles projectiles={projectiles} />
        </div>
      </main>
      <div >
    </div>
  );
}

export default GamePage;

"""
    print(parse_tsx(code))
