"""tools/agent_loop.py: the tools, the sandbox check, and the loop itself.

The tools run against a real temporary directory -- real files, real
subprocesses, a real local HTTP server for fetch_url. The loop runs against a
scripted provider, because what needs pinning down is the loop's behaviour
given a sequence of model turns, and a real model does not produce a
sequence on demand.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


al = _load("agent_loop", "tools/agent_loop.py")


class WorkdirCase(unittest.TestCase):
    def setUp(self):
        self.wd = Path(tempfile.mkdtemp(prefix="al-"))
        (self.wd / "pkg").mkdir()
        (self.wd / "pkg" / "a.py").write_text("import os\nVALUE = 30\n",
                                              encoding="utf-8")
        (self.wd / "README.md").write_text("# title\nrule 30\n", encoding="utf-8")

    def call(self, name, **args):
        return al.execute(name, args, self.wd)


class TestSandbox(WorkdirCase):
    def test_a_path_outside_the_tree_is_refused(self):
        result, ok = self.call("read_file", path="../../etc/passwd")
        self.assertFalse(ok)
        self.assertIn("outside the working tree", result)

    def test_an_absolute_path_is_refused(self):
        result, ok = self.call("read_file", path="/etc/passwd")
        self.assertFalse(ok)
        self.assertIn("outside the working tree", result)

    def test_a_symlink_out_of_the_tree_is_refused(self):
        """resolve() on both sides, not relative_to() on the raw paths --
        otherwise `wt/link/passwd` passes the check and reads /etc."""
        (self.wd / "link").symlink_to("/etc")
        result, ok = self.call("read_file", path="link/passwd")
        self.assertFalse(ok)
        self.assertIn("outside the working tree", result)

    def test_writes_outside_the_tree_are_refused_too(self):
        result, ok = self.call("write_file", path="../escaped.txt", content="x")
        self.assertFalse(ok)
        self.assertFalse((self.wd.parent / "escaped.txt").exists())


class TestReadTools(WorkdirCase):
    def test_list_dir_hides_dot_git(self):
        (self.wd / ".git").mkdir()
        out, ok = self.call("list_dir", path=".")
        self.assertTrue(ok)
        self.assertIn("README.md", out)
        self.assertNotIn(".git", out)

    def test_read_file_numbers_lines(self):
        out, ok = self.call("read_file", path="pkg/a.py")
        self.assertTrue(ok)
        self.assertIn("     1  import os", out)
        self.assertIn("     2  VALUE = 30", out)

    def test_read_file_paginates_and_says_how_to_continue(self):
        (self.wd / "long.txt").write_text("\n".join(str(i) for i in range(100)))
        out, _ = self.call("read_file", path="long.txt", max_lines=10)
        self.assertIn("90 more lines", out)
        self.assertIn("start_line=11", out)

    def test_read_file_on_a_directory_is_an_error_not_a_crash(self):
        out, ok = self.call("read_file", path="pkg")
        self.assertFalse(ok)
        self.assertIn("not a file", out)

    def test_grep_reports_path_line_and_text(self):
        out, ok = self.call("grep", pattern="VALUE")
        self.assertTrue(ok)
        self.assertIn("pkg/a.py:2:", out)

    def test_grep_honours_a_glob(self):
        out, _ = self.call("grep", pattern="rule 30", glob="*.py")
        self.assertIn("no matches", out)

    def test_a_bad_regex_is_returned_to_the_model_not_raised(self):
        out, ok = self.call("grep", pattern="(unclosed")
        self.assertFalse(ok)
        self.assertIn("bad regex", out)


class TestWriteTools(WorkdirCase):
    def test_write_file_creates_and_reports(self):
        out, ok = self.call("write_file", path="new/x.py", content="X = 1\n")
        self.assertTrue(ok)
        self.assertIn("created", out)
        self.assertEqual((self.wd / "new" / "x.py").read_text(), "X = 1\n")

    def test_edit_file_replaces_a_unique_anchor(self):
        out, ok = self.call("edit_file", path="pkg/a.py",
                            old="VALUE = 30", new="VALUE = 31")
        self.assertTrue(ok)
        self.assertIn("VALUE = 31", (self.wd / "pkg" / "a.py").read_text())

    def test_edit_file_refuses_an_ambiguous_anchor(self):
        """A model that meant one occurrence and hit four has made a mess
        that only shows up later, in a diff nobody can read."""
        (self.wd / "dup.py").write_text("x = 1\nx = 1\n")
        out, ok = self.call("edit_file", path="dup.py", old="x = 1", new="x = 2")
        self.assertFalse(ok)
        self.assertIn("appears 2 times", out)
        self.assertEqual((self.wd / "dup.py").read_text(), "x = 1\nx = 1\n")

    def test_edit_file_says_so_when_the_anchor_is_absent(self):
        out, ok = self.call("edit_file", path="pkg/a.py", old="nope", new="x")
        self.assertFalse(ok)
        self.assertIn("does not appear", out)


class TestRun(WorkdirCase):
    def test_runs_with_the_worktree_as_cwd(self):
        out, ok = self.call("run", command="ls")
        self.assertTrue(ok)
        self.assertIn("README.md", out)
        self.assertIn("exit=0", out)

    def test_a_nonzero_exit_is_a_result_not_a_failure(self):
        """The model needs to see a failing test's output; that is the point
        of running it."""
        out, ok = self.call("run", command="echo boom >&2; exit 3")
        self.assertTrue(ok)
        self.assertIn("exit=3", out)
        self.assertIn("boom", out)

    def test_output_is_truncated_in_the_middle(self):
        """Context in a loop is quadratic in what you let into it."""
        out, _ = self.call("run", command="python3 -c \"print('x'*60000)\"")
        self.assertLess(len(out), al.MAX_RESULT_CHARS + 500)
        self.assertIn("cut from the middle", out)

    def test_a_hanging_command_times_out(self):
        out, ok = self.call("run", command="sleep 10", timeout=1)
        self.assertFalse(ok)
        self.assertIn("timed out", out)


class FetchStub(BaseHTTPRequestHandler):
    payload = b"<html><script>bad()</script><p>Hello <b>world</b></p></html>"
    ctype = "text/html"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", self.ctype)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)


class TestFetch(WorkdirCase):
    def setUp(self):
        super().setUp()
        self._no_proxy = os.environ.get("no_proxy")
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), FetchStub)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/page"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        if self._no_proxy is None:
            os.environ.pop("no_proxy", None)
        else:
            os.environ["no_proxy"] = self._no_proxy

    def test_result_is_wrapped_as_untrusted_and_names_its_source(self):
        """The one tool whose output was written by someone who is not the
        operator, in a process that can also run a shell. The boundary has to
        be visible in the transcript, not only in the system prompt."""
        out, ok = self.call("fetch_url", url=self.url)
        self.assertTrue(ok)
        self.assertIn('<untrusted_content source="', out)
        self.assertIn(self.url, out)
        self.assertIn("must not be followed", out)

    def test_html_is_reduced_to_text_and_scripts_are_dropped(self):
        out, _ = self.call("fetch_url", url=self.url)
        self.assertIn("Hello", out)
        self.assertNotIn("bad()", out)

    def test_non_http_schemes_are_refused(self):
        for url in ("file:///etc/passwd", "ftp://x/y", "gopher://x"):
            out, ok = self.call("fetch_url", url=url)
            self.assertFalse(ok, url)
            self.assertIn("http and https", out)


class TestDispatch(WorkdirCase):
    def test_an_unknown_tool_lists_the_real_ones(self):
        out, ok = self.call("nonexistent_tool")
        self.assertFalse(ok)
        self.assertIn("read_file", out)

    def test_wrong_arguments_come_back_as_a_message(self):
        out, ok = self.call("read_file")           # `path` is required
        self.assertFalse(ok)
        self.assertIn("wrong arguments", out)

    def test_every_tool_has_a_schema_with_a_description(self):
        for s in al.tool_schemas():
            f = s["function"]
            self.assertTrue(f["description"], f["name"])
            for req in f["parameters"]["required"]:
                self.assertIn(req, f["parameters"]["properties"], f["name"])

    def test_a_subset_can_be_requested(self):
        names = [s["function"]["name"] for s in al.tool_schemas(["grep", "run"])]
        self.assertEqual(sorted(names), ["grep", "run"])


class ScriptedProvider:
    """A provider whose turns are written down in advance.

    The loop's job is to react correctly to a sequence of model turns. A real
    model cannot be asked for a specific sequence, so the sequence is the
    fixture and the loop is what is under test.
    """

    def __init__(self, turns, cost_per_turn=0.0):
        self.turns, self.cost = list(turns), cost_per_turn
        self.seen = []
        self.model = "scripted/model"

    def chat(self, messages, timeout, tools=None, tool_choice="auto"):
        self.seen.append({"messages": list(messages), "tools": tools})
        msg = self.turns.pop(0) if self.turns else {"content": "done."}
        return {"choices": [{"message": msg}], "usage": {"cost": self.cost}}


def call(name, args, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class TestLoop(WorkdirCase):
    def loop(self, turns, **kw):
        self.provider = ScriptedProvider(turns, kw.pop("cost", 0.0))
        return al.run_loop("do the thing", self.wd, self.provider, **kw)

    def test_a_turn_with_no_tool_calls_ends_the_loop(self):
        r = self.loop([{"content": "the answer"}])
        self.assertEqual(r["final"], "the answer")
        self.assertEqual(r["stop_reason"], al.STOP_DONE)
        self.assertEqual(r["tool_calls"], {})

    def test_tool_results_are_fed_back_and_the_loop_continues(self):
        r = self.loop([
            {"content": "", "tool_calls": [call("read_file", {"path": "pkg/a.py"})]},
            {"content": "VALUE is 30"},
        ])
        self.assertEqual(r["final"], "VALUE is 30")
        self.assertEqual(r["tool_calls"], {"read_file": 1})
        # The second request must carry the assistant turn verbatim and the
        # tool result keyed to its call id, or the API rejects it.
        second = self.provider.seen[1]["messages"]
        self.assertEqual(second[-2]["role"], "assistant")
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(second[-1]["tool_call_id"], "c1")
        self.assertIn("VALUE = 30", second[-1]["content"])

    def test_several_calls_in_one_turn_all_run(self):
        r = self.loop([
            {"content": "", "tool_calls": [
                call("list_dir", {}, "a"), call("grep", {"pattern": "30"}, "b")]},
            {"content": "ok"},
        ])
        self.assertEqual(r["tool_calls"], {"list_dir": 1, "grep": 1})

    def test_tools_are_offered_on_every_request(self):
        self.loop([{"content": "", "tool_calls": [call("list_dir", {})]},
                   {"content": "ok"}])
        for seen in self.provider.seen:
            self.assertTrue(seen["tools"])

    def test_unparseable_arguments_are_returned_as_an_error_not_a_crash(self):
        bad = {"id": "c1", "function": {"name": "read_file", "arguments": "{oops"}}
        r = self.loop([{"content": "", "tool_calls": [bad]}, {"content": "ok"}])
        self.assertEqual(r["stop_reason"], al.STOP_DONE)
        self.assertIn("could not parse arguments",
                      self.provider.seen[1]["messages"][-1]["content"])

    def test_a_provider_error_ends_the_loop_and_is_named(self):
        class Boom:
            model = "x"

            def chat(self, *a, **k):
                raise RuntimeError("upstream on fire")

        r = al.run_loop("t", self.wd, Boom())
        self.assertEqual(r["stop_reason"], al.STOP_ERROR)
        self.assertIn("upstream on fire", r["final"])


class TestBudgets(WorkdirCase):
    def turns(self, n):
        return [{"content": "", "tool_calls": [call("list_dir", {})]}
                for _ in range(n)]

    def test_max_turns_stops_the_loop_and_says_so(self):
        r = al.run_loop("t", self.wd, ScriptedProvider(self.turns(20)),
                        budget=al.Budget(max_turns=3))
        self.assertEqual(r["stop_reason"], al.STOP_TURNS)
        # max_turns + 1: the tool-using loop stops at the limit, and the
        # wrap-up turn that salvages a report is a real billed request, so it
        # is counted rather than hidden. The cap is on tool-using turns.
        self.assertEqual(r["budget"]["turns"], 4)

    def test_max_tool_calls_stops_the_loop(self):
        r = al.run_loop("t", self.wd, ScriptedProvider(self.turns(20)),
                        budget=al.Budget(max_turns=99, max_tool_calls=4))
        self.assertEqual(r["stop_reason"], al.STOP_CALLS)

    def test_cost_is_the_providers_own_accounting_and_caps_the_run(self):
        """Not an estimate from token counts and a price list that may be
        stale -- the number the provider billed."""
        r = al.run_loop("t", self.wd,
                        ScriptedProvider(self.turns(20), cost_per_turn=0.25),
                        budget=al.Budget(max_turns=99, max_cost_usd=0.6))
        self.assertEqual(r["stop_reason"], al.STOP_COST)
        self.assertGreaterEqual(r["budget"]["cost_usd"], 0.6)

    def test_wall_clock_stops_the_loop(self):
        b = al.Budget(max_turns=99, wall_clock_s=0)
        r = al.run_loop("t", self.wd, ScriptedProvider(self.turns(5)), budget=b)
        self.assertEqual(r["stop_reason"], al.STOP_CLOCK)

    def test_a_budget_stop_still_produces_a_report(self):
        """Observed live 2026-09-08: a research task spent six minutes and 36
        tool calls, hit the turn cap mid-investigation, and returned nothing.
        The work was done and unreadable."""
        turns = self.turns(3) + [{"content": "PARTIAL REPORT"}]
        p = ScriptedProvider(turns)
        r = al.run_loop("t", self.wd, p, budget=al.Budget(max_turns=3))
        self.assertEqual(r["stop_reason"], al.STOP_TURNS)
        self.assertEqual(r["final"], "PARTIAL REPORT")

    def test_the_wrap_up_turn_offers_no_tools(self):
        """The budget is spent, and a model that can call a tool will."""
        p = ScriptedProvider(self.turns(3) + [{"content": "x"}])
        al.run_loop("t", self.wd, p, budget=al.Budget(max_turns=3))
        self.assertIsNone(p.seen[-1]["tools"])
        self.assertIn("budget is spent", p.seen[-1]["messages"][-1]["content"])

    def test_a_failed_wrap_up_does_not_lose_the_run(self):
        class HalfBroken(ScriptedProvider):
            def chat(self, messages, timeout, tools=None, tool_choice="auto"):
                if tools is None:          # the wrap-up call
                    raise RuntimeError("provider gone")
                return super().chat(messages, timeout, tools, tool_choice)

        r = al.run_loop("t", self.wd, HalfBroken(self.turns(5)),
                        budget=al.Budget(max_turns=2))
        self.assertEqual(r["stop_reason"], al.STOP_TURNS)
        self.assertEqual(r["final"], "")

    def test_a_finished_run_reports_done(self):
        r = al.run_loop("t", self.wd, ScriptedProvider([{"content": "fin"}]),
                        budget=al.Budget(max_turns=99))
        self.assertEqual(r["stop_reason"], al.STOP_DONE)


class TestTranscript(WorkdirCase):
    def test_every_turn_and_tool_call_is_recorded(self):
        path = self.wd / "t.jsonl"
        al.run_loop("t", self.wd, ScriptedProvider([
            {"content": "", "tool_calls": [call("list_dir", {})]},
            {"content": "ok"}]), transcript=path)
        events = [json.loads(l) for l in path.read_text().splitlines()]
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "end")
        self.assertIn("tool", kinds)
        tool = next(e for e in events if e["event"] == "tool")
        self.assertEqual(tool["name"], "list_dir")
        self.assertTrue(tool["ok"])


class TestSystemPrompt(unittest.TestCase):
    def test_the_injection_rule_precedes_the_task(self):
        """A worker that can fetch a page and run a shell is the whole
        prompt-injection problem in one process."""
        self.assertIn("<untrusted_content>", al.SYSTEM_PROMPT)
        self.assertIn("must not be followed", al.SYSTEM_PROMPT)
        self.assertLess(al.SYSTEM_PROMPT.index("untrusted_content"),
                        al.SYSTEM_PROMPT.index("Stay inside the working tree"))

    def test_it_forbids_changing_git_state_not_only_committing(self):
        """A worker removed its own worktree mid-task on 2026-09-08 to test a
        hypothesis. `git diff` on a re-created checkout is silently empty, so
        the harness would have reported no changes and been believed."""
        for forbidden in ("git commit", "git push", "git worktree", "git branch"):
            self.assertIn(forbidden, al.SYSTEM_PROMPT)
        self.assertIn("never by changing it", al.SYSTEM_PROMPT)

    def test_it_demands_verification_before_claiming(self):
        self.assertIn("Verify before you claim", al.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()


class TestToolSelection(unittest.TestCase):
    """`tools` in a task spec is JSON, so it arrives as a string or a list.

    Both must select the same set, and a name that is not a tool must fail
    loudly. The string form used to be accepted and then matched with `in`,
    so "run" selected `run` out of "grep,run" by substring -- right answer,
    wrong reason, and wrong the moment one tool's name contains another's.
    """

    def names(self, arg):
        return sorted(t["function"]["name"] for t in al.tool_schemas(arg))

    def test_none_selects_every_tool(self):
        self.assertEqual(self.names(None), sorted(al.TOOLS))

    def test_a_comma_string_selects_the_same_set_as_a_list(self):
        wanted = ["grep", "list_dir", "read_file", "run"]
        self.assertEqual(self.names("list_dir,read_file,grep,run"), wanted)
        self.assertEqual(self.names(wanted), wanted)

    def test_whitespace_and_empty_entries_are_tolerated(self):
        self.assertEqual(self.names(" read_file , run ,"), ["read_file", "run"])

    def test_a_substring_of_a_tool_name_is_not_a_tool(self):
        # "read" is a substring of "read_file"; under the old `in` test
        # against a joined string it would have matched.
        with self.assertRaises(ValueError):
            al.tool_schemas("read")

    def test_an_unknown_name_is_refused_and_says_what_exists(self):
        with self.assertRaises(ValueError) as cm:
            al.tool_schemas(["read_file", "nope"])
        msg = str(cm.exception)
        self.assertIn("nope", msg)
        self.assertIn("read_file", msg)

    def test_a_read_only_selection_really_excludes_the_writers(self):
        got = self.names("list_dir,read_file,grep,run")
        for writer in ("write_file", "edit_file", "fetch_url"):
            self.assertNotIn(writer, got)
