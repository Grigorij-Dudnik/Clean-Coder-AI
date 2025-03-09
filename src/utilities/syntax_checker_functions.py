import ast
import yaml
import sass
from lxml import etree
import re
from src.utilities.print_formatters import print_formatted


def check_syntax(file_content, filename):
    parts = filename.split(".")
    extension = parts[-1] if len(parts) > 1 else ''
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
        html_tree = etree.fromstring(html_content, parser)
        significant_errors = [
            error for error in parser.error_log
            # Shut down some error types to be able to parse html from vue
            #if not error.message.startswith('Tag')
            #and "error parsing attribute name" not in error.message
        ]
        if not significant_errors:
            return "Valid syntax"
        else:
            for error in significant_errors:
                return f"HTML line {error.line}: {error.message}"
    except etree.XMLSyntaxError as e:
        return f"Html error occurred: {e}"


def parse_template(code):
    for tag in ['div', 'p', 'span', 'main']:
        function_response = check_template_tag_balance(code, f'<{tag}', f'</{tag}>')
        if function_response != "Valid syntax":
            return function_response
    return "Valid syntax"


def parse_javascript(js_content):
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
        if code[i:i + open_tag_len] == open_tag and code[i + open_tag_len] in [' ', '>', '\n']:
            opened_tags_count += 1
            i += open_tag_len
        elif code[i:i + close_tag_len] == close_tag:
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


def bracket_balance(code, beginnig_bracket='{', end_bracket='}'):
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
    bracket_response = bracket_balance(code, beginnig_bracket='(', end_bracket=')')
    if bracket_response != "Valid syntax":
        return bracket_response
    bracket_response = bracket_balance(code, beginnig_bracket='[', end_bracket=']')
    if bracket_response != "Valid syntax":
        return bracket_response
    bracket_response = bracket_balance(code, beginnig_bracket='{', end_bracket='}')
    if bracket_response != "Valid syntax":
        return bracket_response
    return "Valid syntax"


def parse_scss(scss_code):
    # removing import statements as they cousing error, because function has no access to filesystem
    scss_code = re.sub(r'@import\s+[\'"].*?[\'"];', '', scss_code)
    try:
        sass.compile(string=scss_code)
        return "Valid syntax"
    except sass.CompileError as e:
        return f"CSS/SCSS syntax error: {e}"


# That function does not guarantee finding all the syntax errors in template and script part; but mostly works
def parse_vue_basic(content):
    start_tag_template = re.search(r'<template>', content).end()
    end_tag_template = content.rindex('</template>')
    template = content[start_tag_template:end_tag_template]
    template_part_response = parse_template(template)
    if template_part_response != "Valid syntax":
        return template_part_response

    try:
        script = re.search(r'<script[^>]*>(.*?)</script>', content, re.DOTALL).group(1)
    except AttributeError:
        return "Script part has no valid open/closing tags."
    script_part_response = check_bracket_balance(script)
    if script_part_response != "Valid syntax":
        return script_part_response

    style_match = re.search(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)
    if style_match:
        css = style_match.group(1)
        if css:     # if for the case of empty css block
            style_part_response = parse_scss(style_match.group(1))
            if style_part_response != "Valid syntax":
                return style_part_response

    return "Valid syntax"


# function works, but not used by default as there could be problems with esprima installation
def parse_javascript_esprima(js_content):
    import esprima
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
    eslint_config_path = '.eslintrc.js'
    temp_file_path = "dzik.vue"
    # Create a temporary file
    with open(temp_file_path, 'w', encoding='utf-8') as file:
        file.write(code_string)
    try:
        # Run ESLint on the temporary file
        result = subprocess.run(['D:\\NodeJS\\npx.cmd', 'eslint', '--config', eslint_config_path, temp_file_path, '--fix'], check=True, text=True, capture_output=True)
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
import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

// Scale indicator component showing agreement levels from 1-5
const ScaleIndicator = () => (
  <div className="flex flex-col items-center mb-12">
    <div className="w-full max-w-2xl mx-auto">
      <div className="flex justify-between text-sm text-gray-600 mb-1">
        <span>Highly disagree</span>
        <span>Highly agree</span>
      </div>
      <div className="relative w-full h-[2px] bg-gray-200 mb-8">
        {Array.from({ length: 5 }, (_, i) => i + 1).map((num) => (
          <div
            key={num}
            className="absolute -translate-x-1/2"
            style={{ left: `${((num - 1) * 100) / 4}%` }}
          >
            <div className="absolute -top-3 w-[2px] h-[6px] bg-gray-300" />
            <div className="absolute top-4 text-sm text-gray-600">
              {num}
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

function NavHeader() {
  const router = useRouter();
  return (
    <div className="flex flex-col items-center">
      <div className="flex items-center justify-start w-full mb-2">
        <button
          className="text-gray-700 hover:text-gray-900 mr-4"
          onClick={() => router.back()}
          aria-label="Go back"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
            className="w-5 h-5"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
          </svg>
        </button>
        <h1 className="flex-grow text-center text-xl font-bold">Survey Results</h1>
      </div>
    </div>
  );
}

export default function Page({ params }: { params: Promise<{ uuid: string }> }) {
  const { uuid } = React.use(params);
  const [profile, setProfile] = useState<any>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/profile/${uuid}`
        );
        if (!response.ok) {
          throw new Error('Failed to fetch profile data');
        }
        const data = await response.json();
        setProfile(data);
      } catch (err: any) {
        console.error('Error details:', err);
        setError(err.message || 'An error occurred');
      }
    };

    fetchProfile();
  }, [uuid]);

  if (error) {
    return (
      <div className="p-4 text-red-500">
        {error}
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="p-4">
        Loading profile data...
      </div>
    );
  }

  if (!profile.survey_data) {
    return <div className="p-4 text-gray-700">No survey data available.</div>;
  }

  return (
    <div className="px-4 py-2 text-gray-900 max-w-4xl mx-auto">
      <div className="mb-8">
        <NavHeader />
        <ScaleIndicator />
      </div>

      {profile.survey_data.map((category) => (
        <div key={category.name} className="mb-10">
          <h2 className="text-base font-semibold text-gray-800 mb-4">
            {category.name}
          </h2>
          {category.statements.map((statement: any) => (
            <div key={statement.id} className="flex items-start py-4 border-b border-gray-200 last:border-b-0">
              <span className="text-2xl font-semibold text-gray-900 w-8 text-center">
                {statement.value}
              </span>
              <p className="text-base text-gray-700 leading-relaxed flex-1 ml-6">
                {statement.text}
              </p>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
"""
    print(parse_tsx(code))