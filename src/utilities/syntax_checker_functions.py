import ast
import yaml
import sass
from lxml import etree
import re
import bleach
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

    script = bleach.clean(content, tags=[], strip=True)
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
"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import ProfileCard from "./components/ProfileCard";
import PopupNotification from "./components/PopupNotification";

interface ProfileItem {
  uuid: string;
  full_name: string;
  short_bio?: string;
  bio?: string;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<'Explore' | 'Received' | 'Sent' | 'Matches'>('Explore');
  const [exploreItems, setExploreItems] = useState<ProfileItem[]>([]);
  const [receivedItems, setReceivedItems] = useState<ProfileItem[]>([]);
  const [sentItems, setSentItems] = useState<ProfileItem[]>([]);
  const [matchedItems, setMatchedItems] = useState<ProfileItem[]>([]);
  const [error, setError] = useState('');
  const [notification, setNotification] = useState<{ message: string, type: 'positive' | 'negative' } | null>(null);
  const [loading, setLoading] = useState(false);
  const [iconLoading, setIconLoading] = useState(true);
  const [skip, setSkip] = useState(0);
  const [limit] = useState(10);
  const [totalExploreItems, setTotalExploreItems] = useState(0);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const router = useRouter();

  function goToProfile(uuid: string) {
    const userRole = localStorage.getItem('role');
    if (userRole === "intern") {
      router.push(`/campaign/${uuid}`);
    } else {
      router.push(`/intern/${uuid}`);
    }
  }

  async function handleConnect(uuid: string) {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/invitations/create/${uuid}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error('Failed to create invitation');
      setNotification({ message: 'Invitation sent successfully', type: 'positive' });

      // Optimistically update the explore list
      setExploreItems((prevItems) => prevItems.filter(item => item.uuid !== uuid));
    } catch (err: any) {
      setNotification({ message: err.message, type: 'negative' });
    } finally {
      setLoading(false);
      setTimeout(() => setNotification(null), 3000);
    }
  }

  async function handleAccept(invitationId: string) {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/invitations/accept/${invitationId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error('Failed to accept invitation');
      setNotification({ message: 'Invitation accepted successfully', type: 'positive' });
      setReceivedItems((prevItems) => prevItems.filter(item => item.invitation_id !== invitationId));
    } catch (err: any) {
      setNotification({ message: err.message, type: 'negative' });
    } finally {
      setLoading(false);
      setTimeout(() => setNotification(null), 3000);
    }
  }

  async function handleReject(invitationId: string) {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/invitations/reject/${invitationId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error('Failed to reject invitation');
      setNotification({ message: 'Invitation rejected successfully', type: 'positive' });
      setReceivedItems((prevItems) => prevItems.filter(item => item.invitation_id !== invitationId));
    } catch (err: any) {
      setNotification({ message: err.message, type: 'negative' });
    } finally {
      setLoading(false);
      setTimeout(() => setNotification(null), 3000);
    }
  }

  async function handleCancel(invitationId: string) {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/invitations/cancel/${invitationId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error('Failed to cancel invitation');
      setNotification({ message: 'Invitation canceled successfully', type: 'positive' });
      setSentItems((prevItems) => prevItems.filter(item => item.invitation_id !== invitationId));
    } catch (err: any) {
      setNotification({ message: err.message, type: 'negative' });
    } finally {
      setLoading(false);
      setTimeout(() => setNotification(null), 3000);
    }
  }
  async function fetchExplore() {
    try {
      const userRole = localStorage.getItem('role');
      const token = localStorage.getItem('token');
      
      if (!token) {
        throw new Error('Authentication token not found');
      }

      if (!userRole) {
        throw new Error('User role not found');
      }

      const url = `${process.env.NEXT_PUBLIC_API_URL}${
        userRole === "intern"
          ? '/fetch-campaigns-for-main-page'
          : '/fetch-interns-for-main-page'
      }?skip=${skip}&limit=${limit}`;

      console.log('Fetching from URL:', url);
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch explore items');
      }

      const data = await response.json();
      setExploreItems(prev => [...prev, ...(data.items || [])]);
      setTotalExploreItems(data.total || 0);
    } catch (err: any) {
      console.error('Fetch error:', err);
      setError(err.message);
      setTimeout(() => setError(''), 3000);
    }
  }

  async function fetchReceived() {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/invitations/received`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error('Failed to fetch received invitations');
      const data = await response.json();
      setReceivedItems(data.items || []);
    } catch (err: any) {
      setError(err.message);
      setTimeout(() => setError(''), 3000);
    }
  }

  async function fetchSent() {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/invitations/sent`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error('Failed to fetch sent invitations');
      const data = await response.json();
      setSentItems(data.items || []);
    } catch (err: any) {
      setError(err.message);
      setTimeout(() => setError(''), 3000);
    }
  }
  async function fetchMatches() {
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/matches`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error('Failed to fetch matches');
      const data = await response.json();
      setMatchedItems(data.items || []);
    } catch (err: any) {
      setError(err.message);
      setTimeout(() => setError(''), 3000);
    }
  }


  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setError('Please login first');
      return;
    }
    // Initial load of the first page
    fetchExplore().then(() => setSkip(prev => prev + limit));
  }, []);

  // Infinite scroll: Observe the sentinel at the bottom of the Explore list
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      const [entry] = entries;
      // If sentinel is in view and we have more items to fetch
      if (entry.isIntersecting && skip < totalExploreItems) {
        // Fetch the next batch
        fetchExplore().then(() => {
          setSkip(prev => prev + limit);
        });
      }
    });

    if (sentinelRef.current) {
      observer.observe(sentinelRef.current);
    }

    // Cleanup
    return () => {
      if (sentinelRef.current) {
        observer.unobserve(sentinelRef.current);
      }
    };
  }, [skip, totalExploreItems, limit]);

  const handleTabClick = (tab: 'Explore' | 'Received' | 'Sent' | 'Matches') => {
    setActiveTab(tab);
    if (tab === 'Explore') fetchExplore();
    if (tab === 'Received') fetchReceived();
    if (tab === 'Sent') fetchSent();
    if (tab === 'Matches') fetchMatches();
  };

  let listToRender: ProfileItem[] = [];
  if (activeTab === 'Explore') listToRender = exploreItems;
  if (activeTab === 'Received') listToRender = receivedItems;
  if (activeTab === 'Sent') listToRender = sentItems;
  if (activeTab === 'Matches') listToRender = matchedItems;
  return (
    <main className="max-w-2xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
      <header className="flex items-center justify-between mb-8">
          <button className="w-10 h-10 rounded-full bg-[#EEEEEE] flex items-center justify-center">
            <Image 
              src="/profile.svg" 
              alt="Profile" 
              width={24} 
              height={24} 
              onError={(e) => e.currentTarget.src = '/fallback-icon.svg'} // Fallback icon
            />
          </button>
        <h1 className="text-xl font-semibold">Glovn</h1>
      </header>

      <nav className="flex p-1 mb-8 justify-center bg-[#F5F5F5]/40 rounded-full max-w-md mx-auto">
        {["Explore", "Received", "Sent", "Matches"].map((tab) => (
          <button
            key={tab}
            onClick={() => handleTabClick(tab as 'Explore' | 'Received' | 'Sent' | 'Matches')}
            className={`flex-1 px-6 py-2.5 rounded-full text-sm transition-all duration-300 ${
              activeTab === tab
                ? "bg-white font-medium text-black shadow-sm text-[15px]"
                : "text-gray-400/80 hover:text-gray-500"
            }`}
          >
            {tab}
          </button>
        ))}
      </nav>

      {error && (
        <div className="mb-6 py-2 px-4 w-full text-center bg-[#FFF2F2] text-[#FF0000] text-[14px] rounded-full">
          {error}
        </div>
      )}

      <section className="space-y-4">
        {listToRender.length === 0 ? (
          <div className="text-center py-4 text-gray-500">No items found</div>
        ) : (
          listToRender.map((item) => (
            <ProfileCard
              key={item.uuid}
              item={item}
              onConnect={handleConnect}
              onAccept={handleAccept}
              onReject={handleReject}
              onCancel={handleCancel}
              activeTab={activeTab}
            />
          ))
        )}
      </section>

      {/*<div ref={sentinelRef} style={{ height: "1px" }} />*/}
      {notification && (
        <PopupNotification
          message={notification.message}
          type={notification.type}
          onClose={() => setNotification(null)}
        />
      )}
    </main>
  );
}

"""
    print(parse_tsx(code))