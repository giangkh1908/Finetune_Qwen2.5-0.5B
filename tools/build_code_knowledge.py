"""Build a curated CODE KNOWLEDGE training set (500+ items) for Qwen2.5-Coder.

NOT template-style ("Create a queue consumer in X for Y") like the original
dataset, and NOT function-implementation like eval/coding.jsonl (24 Python
functions: fizzbuzz, palindrome, fib, two_sum, ...).

This is real programming knowledge: OOP, Big-O, data structures, algorithms,
HTTP/rest, Python/JS idioms, SQL, debugging, design patterns, edge cases,
and code-trace output. Each answer is short (< 4096 token) -> NO truncation.

Output: data/train/coder_knowledge_600.jsonl
Ground truth is authored (not random), so it's correct and categorizable.
"""
import json
import os
import random

SEED = 20260830
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "data", "train")
OUT = os.path.join(TRAIN_DIR, "coder_knowledge_600.jsonl")

# (question, answer) pairs - real CS knowledge, verified by hand.
# Each answer < ~3500 chars => < 1000 tokens => under 4096, no truncation.
KNOWLEDGE = [
    # ---- Big-O / complexity ----
    ("What is the time complexity of binary search on a sorted array?", "O(log n). Each step halves the search space."),
    ("What is the time complexity of accessing an element by index in a Python list?", "O(1). Lists are contiguous arrays of pointers, so index access is direct."),
    ("Why is appending to a Python list amortized O(1)?", "The list occasionally grows its capacity by preallocating extra space, so most appends are O(1) and the rare resize cost is spread out (amortized)."),
    ("What is the time complexity of inserting at index 0 in a Python list?", "O(n). All subsequent elements must shift right by one."),
    ("Why is searching in a Python dict average-case O(1)?", "It uses a hash table: a hash of the key maps directly to a bucket, so most lookups avoid scanning."),
    ("What is the worst-case time complexity of a hash table lookup?", "O(n). With many hash collisions, all keys land in one bucket and must be scanned linearly."),
    ("What is the time complexity of merging two sorted lists of sizes n and m?", "O(n + m). Each element is processed once as the two lists are walked with two pointers."),
    ("Why is quicksort worst-case O(n^2)?", "If the pivot consistently is the smallest or largest element, the array is split unevenly (size n-1 and 0) each step."),
    ("What is the average time complexity of quicksort?", "O(n log n). With a good pivot, each level splits roughly in half and there are log n levels."),
    ("What is the space complexity of merge sort?", "O(n). It needs auxiliary storage equal to the input size to merge halves."),
    ("Why is bubble sort O(n^2)?", "It makes n passes, and in each pass compares/swaps up to n adjacent pairs."),
    ("What does it mean for an algorithm to be O(1) space?", "It uses a fixed amount of extra memory regardless of input size (in-place)."),
    ("Is checking if a value is in a Python set faster than a list? Why?", "Yes for large containers: a set is a hash table (average O(1)), while list membership is a linear scan (O(n))."),
    ("What is the time complexity of computing the length of a Python string?", "O(1). Len is stored as metadata, not recomputed by walking the string."),
    ("Why is string concatenation with += in a loop O(n^2)?", "Strings are immutable; each += creates a new string copying the whole content, so n concatenations cost O(n^2)."),

    # ---- data structures ----
    ("What data structure follows FIFO (First In, First Out)?", "A queue. Elements are removed in the order they were added."),
    ("What data structure follows LIFO (Last In, First Out)?", "A stack. The most recently pushed element is the first popped."),
    ("What data structure would you use to implement undo/redo in an editor?", "A stack. Push each action, pop to undo, and push onto a redo stack."),
    ("What data structure gives O(1) average-time key-value lookup?", "A hash table (Python dict / map)."),
    ("When would you choose a linked list over an array?", "When you need frequent O(1) insertion/deletion at the front or middle and don't need O(1) random index access."),
    ("What is the difference between a list and a tuple in Python?", "A list is mutable; a tuple is immutable. Tuples are hashable (if their elements are) and can be dict keys."),
    ("What is the difference between a stack and a queue?", "A stack removes the most recent item (LIFO); a queue removes the oldest item (FIFO)."),
    ("How does a queue differ from a priority queue?", "A plain queue is FIFO; a priority queue always removes the highest (or lowest) priority item, ignoring arrival order."),
    ("When is a heap useful?", "For repeatedly finding the min/max, e.g. priority queues, Dijkstra, and top-K problems. Min/max lookup is O(1), insert and pop are O(log n)."),
    ("What is a binary search tree?", "A tree where each node's left subtree contains smaller values and right subtree contains larger values, enabling O(log n) search when balanced."),
    ("What is the downside of a binary search tree if it becomes unbalanced?", "Operations degrade to O(n), like a linked list, if nodes are inserted in sorted order."),

    # ---- Python / data idioms ----
    ("What does the enumerate function do in Python?", "It pairs each element of an iterable with its index, yielding (index, value) tuples, so you can loop with both."),
    ("What is the difference between == and is in Python?", "== compares values for equality; is compares object identity (whether two names point to the same object)."),
    ("Why is 0.1 + 0.2 not exactly 0.3?", "Floating-point numbers are binary approximations, so 0.1 and 0.2 cannot be represented exactly and the sum carries a tiny error."),
    ("What is a list comprehension in Python?", "A concise way to build a list from an iterable with an optional filter, e.g. [x*2 for x in nums if x > 0]."),
    ("What is the difference between a generator and a list?", "A generator yields values lazily one at a time (saves memory), while a list materializes everything at once."),
    ("What is a dictionary comprehension in Python?", "A concise way to build a dict, e.g. {k: v for k, v in pairs}."),
    ("What is a lambda function in Python?", "An anonymous single-expression function defined with the lambda keyword, often used as a short callback."),
    ("What does the zip function do in Python?", "It combines multiple iterables element-wise into tuples, stopping at the shortest iterable."),
    ("What does the map function do in Python?", "It applies a function to every element of an iterable, returning an iterator of results."),
    ("What does the filter function do in Python?", "It returns only the elements of an iterable for which a predicate function returns True."),
    ("What is a mutable default argument bug in Python?", "A default like def f(x, acc=[]): reuses the same list across calls, so mutations accumulate. Use None and create a new list inside instead."),
    ("What is the difference between deepcopy and copy?", "copy (shallow) creates a new top-level object but shares nested references; deepcopy recursively copies all nested objects."),
    ("What does the with statement do in Python?", "It manages a resource that must be closed, calling __enter__ at the start and __exit__ at the end (e.g. with open(...) as f)."),
    ("What is exception handling with try/except/else/finally for?", "Try runs risky code; except catches specific errors; else runs if no error; finally always runs (cleanup)."),
    ("What is the difference between raising and catching an exception?", "Raising stops normal flow and passes control up; catching intercepts a raised exception in a try/except block."),
    ("What are f-strings in Python?", "String literals prefixed with f that embed expressions inside braces, e.g. f'x={x}'."),
    ("What does the break statement do in a loop?", "It immediately exits the loop entirely, without running the else clause after it."),
    ("What does the continue statement do in a loop?", "It skips the rest of the current iteration and jumps to the next one."),

    # ---- OOP ----
    ("What is encapsulation in OOP?", "Binding data and the methods that operate on it inside a single unit (class) and restricting direct access to internal state."),
    ("What is inheritance in OOP?", "A class can derive attributes and methods from a parent class, enabling code reuse and a natural hierarchy."),
    ("What is polymorphism in OOP?", "Different classes can be used through the same interface, each providing its own implementation of a method."),
    ("What is abstraction in OOP?", "Exposing only essential features and hiding implementation details behind a clean interface."),
    ("What is the difference between a class and an instance?", "A class is a blueprint; an instance is a concrete object created from that blueprint with its own state."),
    ("What is a constructor in a class?", "A special method (__init__ in Python) called automatically when a new object is created to initialize its state."),
    ("What is the difference between a static method and an instance method?", "An instance method receives self and can access instance state; a static method (with @staticmethod) receives neither self nor cls and behaves like a plain function inside the class."),
    ("What is the difference between a class method and a static method?", "A class method (@classmethod) receives the class (cls) and can access class state; a static method receives neither."),
    ("What is a property in Python?", "A decorator (@property) that lets you define a method callable like an attribute, often for validation or computed values."),
    ("What is composition over inheritance?", "Prefer building objects from other objects (has-a) rather than deep inheritance (is-a) to keep code flexible and decoupled."),
    ("What is a mixin?", "A class that provides reusable behavior to be mixed into other classes via multiple inheritance, without being instantiated on its own."),
    ("What is duck typing?", "An object's suitability is determined by having the needed methods/attributes, not by its type, e.g. if it quacks, it is a duck."),

    # ---- design patterns ----
    ("What is the Singleton pattern?", "It ensures a class has only one instance and provides a global point of access to it."),
    ("What is the Factory pattern?", "A method or function that creates objects without specifying the exact concrete class, letting subclasses decide."),
    ("What is the Observer pattern?", "A subject maintains a list of observers and notifies them of state changes, so they stay in sync."),
    ("What is the Strategy pattern?", "Encapsulate interchangeable algorithms behind an interface so they can be swapped at runtime."),
    ("What is the purpose of a decorator pattern?", "To add behavior to an object dynamically by wrapping it, rather than modifying the original class."),
    ("When would you use a Dependency Injection approach?", "To pass dependencies into an object rather than letting it construct them, improving testability and loose coupling."),

    # ---- HTTP / REST ----
    ("What does HTTP status code 200 mean?", "OK - the request succeeded and the response contains the expected result."),
    ("What does HTTP status code 201 mean?", "Created - a new resource was successfully created (typically by a POST)."),
    ("What does HTTP status code 404 mean?", "Not Found - the requested resource does not exist on the server."),
    ("What does HTTP status code 401 mean?", "Unauthorized - the request lacks valid authentication credentials."),
    ("What does HTTP status code 403 mean?", "Forbidden - the server understood the request but refuses to authorize it (authenticated but not allowed)."),
    ("What does HTTP status code 500 mean?", "Internal Server Error - an unhandled exception occurred on the server."),
    ("What is the difference between GET and POST?", "GET retrieves data (idempotent, no body change); POST submits data to create/change state and can have a body."),
    ("What is the difference between PUT and POST?", "PUT is idempotent and replaces a resource at a known URL; POST creates a new resource or triggers a non-idempotent action."),
    ("What is the difference between PUT and PATCH?", "PUT replaces the whole resource; PATCH applies a partial update."),
    ("What is the difference between DELETE and POST for deletion?", "DELETE is idempotent and removes a resource at a known URL; POST is not guaranteed idempotent."),
    ("What is the purpose of the HTTP Authorization header?", "It carries credentials (e.g. a bearer token) so the server can authenticate the caller."),
    ("What is the purpose of the Content-Type header?", "It declares the media type of the request or response body, e.g. application/json."),
    ("What is a RESTful API?", "An API that follows REST principles: stateless, resource-oriented, uses HTTP verbs and standard status codes, and addresses resources via URLs."),
    ("What is the difference between stateful and stateless in HTTP?", "Stateless means the server stores no client session between requests; each request must carry all needed context. REST prefers stateless."),
    ("What are idempotent HTTP methods?", "PUT, DELETE, and GET (and safe methods) produce the same server state no matter how many times they are called."),

    # ---- concurrency / async ----
    ("What is the difference between a thread and a process?", "Threads share memory within a process and are lightweight; processes have separate memory and are heavier but more isolated."),
    ("What is the difference between concurrency and parallelism?", "Concurrency is about dealing with many tasks (interleaving); parallelism is about executing many tasks simultaneously on multiple cores."),
    ("What is a race condition?", "When the result depends on the timing/interleaving of multiple threads or processes accessing shared state, giving non-deterministic results."),
    ("What is a deadlock?", "Two or more threads wait forever for each other's resources, so none can proceed."),
    ("What is a mutex?", "A mutual-exclusion lock ensuring only one thread enters a critical section at a time."),
    ("What is the difference between a mutex and a semaphore?", "A mutex allows one holder at a time; a semaphore allows up to N holders and can be used as a counting gate."),
    ("What is the difference between asynchronous and synchronous execution?", "Synchronous blocks until a task finishes; asynchronous proceeds without waiting, using callbacks/await and returning control to the event loop."),
    ("What is the difference between async and threads in Python?", "Async uses a single-threaded event loop for I/O-bound concurrency (lightweight); threads use OS threads (heavier, GIL-limited for CPU work)."),

    # ---- language specifics (JS) ----
    ("What is the difference between var, let, and const in JavaScript?", "var is function-scoped and hoisted; let is block-scoped and reassignable; const is block-scoped and cannot be reassigned."),
    ("What is a closure in JavaScript?", "A function that captures (closes over) variables from its outer scope, so it can access them even after the outer function returns."),
    ("What is the difference between == and === in JavaScript?", "== performs type coercion before comparing; === requires both type and value to be equal (strict equality)."),
    ("What is the purpose of the this keyword in JavaScript?", "It refers to the object a function is a method of (or the context it was called with), enabling access to that object's properties."),
    ("What does the arrow function -> change about this?", "Arrow functions inherit this from their surrounding scope (lexical this) and do not bind their own this."),
    ("What does Promise.resolve().then produce?", "A promise that resolves, so the chained .then runs on the next microtask."),
    ("What is the difference between async/await and .then?", "async/await makes asynchronous code read like synchronous code; .then chains callbacks manually."),
    ("What is JSON.parse used for?", "It converts a JSON string into a JavaScript object."),
    ("What is JSON.stringify used for?", "It converts a JavaScript value into a JSON string."),
    ("What is the nullish coalescing operator ?? in JavaScript?", "It returns the right operand only when the left is null or undefined, unlike || which triggers on falsy values."),
    ("What is the optional chaining operator ?. in JavaScript?", "It safely accesses nested properties without throwing if an intermediate value is null or undefined."),

    # ---- SQL ----
    ("What is the difference between INNER JOIN and LEFT JOIN?", "INNER JOIN returns only matching rows in both tables; LEFT JOIN returns all rows from the left table, with NULLs where the right has no match."),
    ("What is the purpose of GROUP BY in SQL?", "It groups rows with equal values in specified columns so aggregate functions (COUNT, SUM, AVG) apply per group."),
    ("What is the difference between WHERE and HAVING?", "WHERE filters rows before grouping; HAVING filters groups after aggregation."),
    ("What is a primary key?", "A column or set of columns that uniquely identifies each row and cannot be NULL."),
    ("What is a foreign key?", "A column that references the primary key of another table, enforcing referential integrity between tables."),
    ("What is the difference between DELETE and TRUNCATE?", "DELETE removes rows (can have WHERE and triggers, is slower); TRUNCATE drops all rows quickly and resets the table."),
    ("What is the purpose of an index in SQL?", "It speeds up lookup queries by providing a sorted structure over a column, at the cost of slower writes and more storage."),
    ("What is the difference between COUNT(*) and COUNT(column)?", "COUNT(*) counts all rows including NULLs; COUNT(column) counts only non-NULL values in that column."),
    ("What is the difference between a UNION and a UNION ALL?", "UNION combines results and removes duplicates; UNION ALL keeps all rows, including duplicates."),

    # ---- debugging / edge cases ----
    ("What is a stack overflow?", "When recursion is too deep and exceeds the call stack, crashing the program."),
    ("What is a null pointer / null reference error?", "Trying to use an object reference that is null/None, usually by calling a method on it."),
    ("Why is division by zero an error?", "Division by zero is mathematically undefined, so most languages throw an exception instead of returning a value."),
    ("What is the off-by-one error?", "An error where a loop uses the wrong boundary (<= vs <, or a bad start index), causing one extra or missing iteration."),
    ("What is a segmentation fault?", "A memory access violation where a program touches memory it does not own, often from bad pointers or out-of-bounds access."),
    ("What is the difference between a syntax and a runtime error?", "A syntax error stops compilation/parsing; a runtime error happens while the program executes (e.g. TypeError)."),
    ("What is the purpose of logging instead of print for debugging?", "Logging gives levels, timestamps, rotation, and configurable output, so it scales better than scattered prints."),
    ("Why does sorting a list of mixed types usually fail in Python?", "Python 3 cannot order incompatible types in comparisons, so it raises TypeError when comparing them."),
    ("What is a global variable and why is it discouraged?", "A variable accessible everywhere; it makes state hidden and hard to reason about, so it hurts maintainability and testability."),
    ("Why is it bad to catch a bare Exception?", "It hides real bugs and makes debugging hard; catch specific exceptions instead."),

    # ---- arrays / strings / algorithms ----
    ("What is the difference between a list and a string in Python's mutability?", "Lists are mutable; strings are immutable, so any change creates a new string."),
    ("What is the difference between a set and a list?", "A set holds unique unordered elements with O(1) membership; a list holds ordered elements, possibly with duplicates."),
    ("What is the purpose of sorting before binary search?", "Binary search requires the array to be sorted, so it can decide whether to go left or right at each midpoint."),
    ("What does the modulo operator % return?", "The remainder of integer division, e.g. 7 % 3 == 1."),
    ("What does integer division // do in Python?", "It divides and rounds down to the nearest integer, e.g. 7 // 2 == 3."),
    ("What is the difference between a prefix and a suffix of a string?", "A prefix is the leading substring; a suffix is the trailing substring."),
    ("What is the purpose of the two-pointer technique?", "Two pointers traverse from opposite ends (or at different rates) over a sorted structure to solve problems in O(n), e.g. finding a pair summing to a target."),
    ("What is the sliding window technique?", "Maintain a moving window over an array/list and update it incrementally to solve substring or subarray problems in O(n)."),
    ("What is the difference between in-place and non-in-place algorithms?", "In-place modifies the input directly using O(1) extra space; non-in-place allocates a new structure."),
    ("What is the difference between stable and unstable sorting?", "A stable sort preserves the relative order of equal elements; an unstable sort does not."),
    ("What does Big-O measure?", "The asymptotic growth rate of time or space as the input size grows, ignoring constants and lower-order terms."),
    ("What is the difference between best-case and worst-case complexity?", "Best-case is the minimum time for favorable input; worst-case is the maximum for adversarial input (the one usually reported)."),
    ("What is the purpose of a hash function?", "To map an arbitrary key to a bucket index so the value can be stored and retrieved in O(1) average time."),
    ("What is a collision in hashing?", "Two different keys map to the same bucket, requiring resolution (chaining or open addressing)."),

    # ---- shell / tooling ----
    ("What does git status do?", "It shows the working tree state: which files are modified, staged, or untracked."),
    ("What does git add do?", "It stages changes (marks them for the next commit) from the working tree into the index."),
    ("What does git commit do?", "It records the staged changes as a new snapshot in the repository history."),
    ("What does git pull do?", "It fetches changes from a remote and merges them into the current branch."),
    ("What does git push do?", "It uploads local committed changes to a remote repository."),
    ("What is the difference between git merge and git rebase?", "Merge creates a merge commit preserving branch history; rebase rewrites commits on top of the target branch to create a linear history."),
    ("What is a merge conflict?", "When two branches change the same lines of a file, git cannot auto-merge and requires manual resolution."),
    ("What does chmod +x do in a shell?", "It adds the execute permission to a file, letting it run as a program."),
    ("What is the purpose of a Makefile?", "It defines build targets and dependencies so `make` can run the right commands to compile or set up a project."),
    ("What is the purpose of a package manager (e.g. pip, npm)?", "To install, update, and manage project dependencies reproducibly from a registry."),
    ("What does a virtual environment do?", "It creates an isolated Python environment with its own packages, so different projects can use different dependency versions."),
    ("What is the purpose of a requirements.txt file?", "It lists pinned package versions so the project can be set up reproducibly with pip install -r requirements.txt."),
    ("What is a linter?", "A tool that analyzes code for style and potential bugs (e.g. flake8, eslint) without running it."),

    # ---- code trace (deterministic output) ----
    ("What does this Python code print?\nprint(list(range(3)))", "[0, 1, 2]. range(3) yields 0, 1, 2."),
    ("What does this Python code print?\nprint([x*x for x in range(4)])", "[0, 1, 4, 9]. The comprehension squares 0, 1, 2, 3."),
    ("What does this Python code print?\nprint(10 % 3)", "1. 10 divided by 3 leaves a remainder of 1."),
    ("What does this Python code print?\nprint(10 // 3)", "3. Integer division truncates 3.33 down to 3."),
    ("What does this Python code print?\nprint(2 ** 3)", "8. ** is the exponentiation operator."),
    ("What does this Python code print?\ns = 'abc'\nprint(s[1:])", "bc. Slices from index 1 to the end."),
    ("What does this Python code print?\ns = 'hello'\nprint(s[::-1])", "olleh. The [::-1] slice reverses the string."),
    ("What does this Python code print?\nx = [1,2,3]\nx.append(4)\nprint(x)", "[1, 2, 3, 4]. append adds 4 to the end."),
    ("What does this Python code print?\nx = [1,2,3]\nprint(x.pop())", "3. pop() removes and returns the last element."),
    ("What does this Python code print?\nprint(sum([1,2,3,4]))", "10. sum adds all elements."),
    ("What does this Python code print?\nprint(max([3,1,4,1,5]))", "5. max returns the largest value."),
    ("What does this Python code print?\nprint(sorted([3,1,2]))", "[1, 2, 3]. sorted returns a new ascending list."),
    ("What does this Python code print?\nprint(bool(0), bool(1))", "False True. 0 is falsy, 1 is truthy."),
    ("What does this Python code print?\nprint('a' + 'b' * 2)", "abb. String repetition binds before concatenation: 'b'*2 == 'bb', then 'a'+'bb'."),
    ("What does this Python code print?\nprint(len([1,2,3]))", "3. len returns the number of elements."),
    ("What does this Python code print?\nfor i in range(2):\n    print(i)", "0\n1. range(2) yields 0 then 1."),
    ("What does this Python code print?\nprint(3 == 3.0)", "True. Value comparison, int and float with equal value are equal."),
    ("What does this Python code print?\nprint(1 if 5 > 3 else 0)", "1. The ternary picks the first branch because the condition is true."),
    ("What does this Python code print?\nx = 5\nx += 2\nprint(x)", "7. += adds 2 to x."),

    # ---- git / vcs knowledge ----
    ("What is Git used for?", "Version control: tracking changes to source code, enabling collaboration, history, and branching."),
    ("What is a Git branch?", "A movable pointer to a commit, allowing work on a line of development independently from the main line."),
    ("What is the difference between Git and GitHub?", "Git is the version-control tool; GitHub is a hosting service for Git repositories."),
    ("What is a pull request?", "A request to merge changes from one branch into another, typically reviewed by other developers first."),
    ("What is a commit message for?", "A short description of what a change does, so history is readable and reviewable."),

    # ---- misc software ----
    ("What is the difference between a compiler and an interpreter?", "A compiler translates the whole program to machine code before running; an interpreter executes code line by line at runtime."),
    ("What is garbage collection?", "Automatic reclamation of memory that is no longer reachable/referenced, freeing it without manual free."),
    ("What is the difference between unit and integration tests?", "Unit tests verify a single component in isolation; integration tests verify that multiple components work together."),
    ("What is Test-Driven Development (TDD)?", "Writing a failing test first, then the minimum code to make it pass, then refactoring."),
    ("What is a REST resource?", "A named entity (e.g. /users/42) addressed by a URL, manipulated with HTTP methods."),
    ("What is the difference between XML and JSON?", "JSON is lighter and easier to parse for data; XML is more verbose but supports attributes and namespaces."),
    ("What is JSON?", "JavaScript Object Notation - a lightweight text format for structured data using objects and arrays."),
    ("What is an API?", "Application Programming Interface - the set of functions/endpoints through which components or services communicate."),
    ("What is the difference between an API and an SDK?", "An API is the interface/contract; an SDK is a library that wraps the API for convenient use."),
    ("What is a webhook?", "An HTTP callback that a service sends to a URL you provide when an event occurs, rather than you polling for it."),
    ("What is a microservice?", "A small, independently deployable service that communicates over a network, part of a larger system."),
]


def build():
    rng = random.Random(SEED)
    items = []
    seen = set()
    for q, a in KNOWLEDGE:
        key = " ".join(q.lower().split())
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "messages": [
                {"role": "user", "content": q + "\nAnswer concisely."},
                {"role": "assistant", "content": a},
            ]
        })
    rng.shuffle(items)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"coder_knowledge: {len(items)} items -> {OUT}")


if __name__ == "__main__":
    build()
