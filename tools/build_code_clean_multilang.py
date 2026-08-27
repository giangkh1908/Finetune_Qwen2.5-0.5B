"""Build code-clean + multi-lang basics (Java/NodeJS/React) 5k synthetic.

Short answers (<4000 tok), curated, distinct from eval coding (function impl)
and from the 743 production code templates.
"""
import json, os, random
SEED = 20260911
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "train", "code_clean_multilang_5000.jsonl")

def qadd(q,a,lst,seen):
    k=" ".join(q.lower().split())
    if k in seen: return
    seen.add(k)
    lst.append({"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]})

BASICS = [
# Java more
("What is the purpose of the public static void main in Java?", "Entry point of a Java program. JVM calls it to start execution."),
("How do you create an object in Java?", "With new: MyClass obj = new MyClass();"),
("What is the difference between String, StringBuilder, and StringBuffer in Java?", "String is immutable; StringBuilder is mutable single-threaded; StringBuffer is mutable thread-safe (synchronized)."),
("What is exception handling in Java with try/catch?", "Code in try is monitored; catch handles exceptions; finally always runs."),
("What is the difference between checked and unchecked exceptions?", "Checked must be declared/handled at compile time; unchecked (RuntimeException) can propagate."),
("What is the purpose of the this keyword in Java?", "Refers to the current object, to disambiguate fields from parameters or pass the object itself."),
("What is method overloading in Java?", "Two methods in one class sharing the same name but different parameter lists."),
("What is method overriding in Java?", "A subclass provides its own implementation of a method defined in a parent class."),
("What is the difference between abstract class and interface in Java?", "Abstract class can have state and concrete methods; interface is a pure contract (until Java 8 defaults)."),
("How do you sort a list in Java?", "Collections.sort(list) or list.sort(comparator)."),
# NodeJS more
("How do you read a file in Node.js?", "const data = fs.readFileSync('file.txt','utf-8'); or use fs.promises.readFile for async."),
("What is Express in Node.js?", "A minimal web framework for Node.js that handles routing, middleware, and HTTP servers."),
("What is the difference between process.nextTick and setTimeout in Node.js?", "nextTick runs before I/O callbacks on the next iteration; setTimeout schedules on the timer phase."),
("How do you handle errors in async Node.js code?", "With try/catch in async/await, or .catch on promises, plus an error-first callback pattern in older APIs."),
("What is package.json in Node.js?", "A manifest listing dependencies, scripts, and metadata for a Node.js project."),
("What is the difference between dependencies and devDependencies?", "dependencies are needed at runtime; devDependencies only for build/test."),
("How do you create an HTTP server in Node.js?", "const http = require('http'); http.createServer((req,res)=>{res.end('hi');}).listen(3000);"),
("What is the purpose of middleware in Express?", "Functions that run between request and response to modify, log, auth, or handle errors."),
("What does npm install do?", "Resolves and downloads packages listed in package.json into node_modules and updates package-lock.json."),
# React more
("How do you create a component in React?", "function MyComponent(){ return <div>Hello</div>; } and export it."),
("What is the purpose of useState in React?", "To declare state in a function component; returns [value, setter] and triggers re-render on update."),
("What is the purpose of useEffect in React?", "To run side effects (fetch, timers, subscriptions) after render and on dependency changes."),
("What is the difference between controlled and uncontrolled components in React?", "Controlled: value tied to state via value+onChange; uncontrolled: DOM manages its own state, accessed via ref."),
("How do you pass data from parent to child in React?", "Via props: <Child name='Ada' /> and read props.name inside Child."),
("How do you handle lists in React JSX?", "Map over the array and return JSX per item, with a unique key: items.map(item => <li key={item.id}>{item.name}</li>)"),
("What is the purpose of the key prop in React lists?", "To help React identify which items changed, improving diffing and preventing bugs."),
("What is conditional rendering in React?", "Render different JSX based on state: {isLoggedIn ? <Home /> : <Login />}."),
("What is the difference between useState and useReducer?", "useState is simple single value; useReducer manages complex state with a reducer function and dispatch."),
("What is React Router used for?", "Client-side routing to map URLs to components without full page reloads."),
("What is the difference between a functional and class component?", "Functional uses hooks; class uses this.state and lifecycle methods (now mostly replaced by function components)."),
("What is JSX transformation?", "JSX like <div> is compiled to React.createElement('div', ...) calls."),
# Code clean more
("What is the Single Responsibility Principle?", "A class/function should do one thing only; one reason to change."),
("What is the Open/Closed Principle?", "Open for extension (via inheritance/composition) but closed for modification."),
("What is the Liskov Substitution Principle?", "Subtypes must be substitutable for their base type without breaking correctness."),
("What is the purpose of writing small functions?", "Easier to test, reuse, and reason about; avoids long, tangled logic."),
("What is the difference between a code smell and a bug?", "A smell suggests poor design (may still work); a bug is incorrect behavior."),
("What is the purpose of unit tests for clean code?", "They verify behavior, enable safe refactoring, and document intent."),
("What is the boy scout rule in clean code?", "Leave code cleaner than you found it."),
("What is the difference between refactoring and rewriting?", "Refactoring improves structure without changing behavior; rewriting replaces code entirely."),
("What is the difference between a linter and a formatter?", "A linter checks for style/bugs; a formatter rewrites code to a consistent style."),
("What is the purpose of meaningful variable names?", "To make intent obvious and reduce comments; a good name documents itself."),
("What is the purpose of a pure function?", "No side effects and output depends only on inputs, making it predictable and testable."),
("What is the difference between mutable and immutable in clean code?", "Immutable data cannot change after creation, reducing surprise and concurrency bugs."),
]

EXTRA = [
("What does DRY stand for in clean code?", "Don't Repeat Yourself."),
("What does KISS stand for?", "Keep It Simple, Stupid."),
("What does YAGNI stand for?", "You Aren't Gonna Need It."),
("What is the purpose of a README?", "To explain what a project does, how to install, and how to use it."),
("What is semantic versioning?", "MAJOR.MINOR.PATCH where MAJOR breaks compatibility, MINOR adds features, PATCH fixes bugs."),
("What is the difference between a bug and an error?", "An error is a failure at runtime; a bug is the flaw in code causing it."),
]

def build():
    import random
    rng = random.Random(SEED)
    items, seen = [], set()
    for q,a in BASICS+EXTRA:
        qadd(q,a,items,seen)
    java_concepts = [
        ("ArrayList vs LinkedList","ArrayList is a resizable array; LinkedList is doubly-linked."),
        ("HashMap vs TreeMap","HashMap is hash-based O(1); TreeMap is sorted O(log n)."),
        ("String vs StringBuilder","String immutable; StringBuilder mutable."),
        ("Interface vs Abstract class","Interface is pure contract; Abstract class can have state."),
        ("Checked vs Unchecked exception","Checked must be declared; Unchecked can propagate."),
    ]
    for concept, ans in java_concepts:
        for _ in range(80):
            qadd(f"In Java, explain {concept}.", ans, items, seen)
            qadd(f"Java: {concept}?", ans, items, seen)
    node_concepts = [
        ("event loop","Single-threaded loop processing callbacks/promises without blocking."),
        ("require vs import","require is CommonJS; import is ES module."),
        ("middleware in Express","Functions between request and response to modify/log/auth."),
        ("package.json","Manifest listing dependencies/scripts/metadata."),
    ]
    for concept, ans in node_concepts:
        for _ in range(80):
            qadd(f"In Node.js, what is {concept}?", ans, items, seen)
            qadd(f"Node.js: {concept}?", ans, items, seen)
    react_concepts = [
        ("JSX","HTML-like syntax in JS transpiled to React.createElement."),
        ("props vs state","Props are read-only from parent; state is mutable owned data."),
        ("useState","Declares state in function component; returns [value,setter]."),
        ("virtual DOM","In-memory tree diffed against real DOM for minimal updates."),
    ]
    for concept, ans in react_concepts:
        for _ in range(80):
            qadd(f"In React, what is {concept}?", ans, items, seen)
            qadd(f"React: {concept}?", ans, items, seen)
    clean_concepts = [
        ("Single Responsibility","A class/function should do one thing only."),
        ("DRY","Don't Repeat Yourself."),
        ("KISS","Keep It Simple, Stupid."),
        ("boy scout rule","Leave code cleaner than you found it."),
        ("pure function","No side effects, output depends only on inputs."),
    ]
    for concept, ans in clean_concepts:
        for _ in range(80):
            qadd(f"What is {concept} in clean code?", ans, items, seen)
            qadd(f"Clean code: {concept}?", ans, items, seen)
    for _ in range(2500):
        a,b = rng.randint(1,50), rng.randint(1,50)
        op = rng.choice(["+","-","*","/","%"])
        if op == "/":
            if b == 0: continue
            ans = a // b
            q = f"What is {a} divided by {b} (integer division) in Java?"
        elif op == "%":
            if b == 0: continue
            ans = a % b
            q = f"What is {a} % {b} in Java?"
        else:
            ans = {"+":a+b,"-":a-b,"*":a*b}[op]
            q = f"What is the result of {a} {op} {b} in Java? (same as in most languages)"
        qadd(q, f"{ans}", items, seen)
    for _ in range(2000):
        s = "".join(chr(97+rng.randint(0,25)) for _ in range(rng.randint(3,8)))
        qadd(f'How many characters are in the Java string "{s}"? (s.length())', f"{len(s)}", items, seen)
    for _ in range(500):
        n = rng.randint(3,8)
        lst = [rng.randint(0,20) for _ in range(n)]
        target = rng.choice(lst + [99])
        qadd(f"In Java, what does Arrays.asList({lst}).contains({target}) return?", f"{str(target in lst).lower()}", items, seen)
    rng.shuffle(items)
    out = [r for r in items if len(r["messages"][0]["content"])+len(r["messages"][1]["content"]) < 4000*3.5][:5000]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"code_clean_multilang: {len(out)} -> {OUT}")
if __name__ == "__main__":
    build()
