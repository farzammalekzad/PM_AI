'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class TextNode {
  constructor(value) {
    this.value = String(value);
  }

  get textContent() {
    return this.value;
  }
}

class Element {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.className = '';
    this.dataset = {};
    this.dir = '';
    this._text = '';
    this.classList = {
      add: (...names) => {
        const namesPresent = new Set(this.className.split(/\s+/).filter(Boolean));
        names.forEach(name => namesPresent.add(name));
        this.className = [...namesPresent].join(' ');
      },
    };
  }

  append(...nodes) {
    this._text = '';
    nodes.forEach(node => {
      node.parentNode = this;
    });
    this.children.push(...nodes);
  }

  replaceChildren(...nodes) {
    this._text = '';
    nodes.forEach(node => {
      node.parentNode = this;
    });
    this.children = [...nodes];
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }

  get textContent() {
    return this._text + this.children.map(child => child.textContent).join('');
  }

  get childElementCount() {
    return this.children.filter(child => child instanceof Element).length;
  }
}

const document = {
  createElement: tagName => new Element(tagName),
  createTextNode: value => new TextNode(value),
};
const source = fs.readFileSync(path.join(__dirname, '..', 'app', 'web', 'static', 'app.js'), 'utf8');
const start = source.indexOf('function appendPlain');
const end = source.indexOf('async function models');
assert.notEqual(start, -1, 'Markdown helpers must be present');
assert.notEqual(end, -1, 'Markdown helper boundary must be present');
const context = {document, Set};
vm.runInNewContext(source.slice(start, end), context);

const bubble = document.createElement('div');
context.renderMarkdown(bubble, [
  '# عنوان پاسخ',
  '',
  'متن **پررنگ** و *مورب* با `code` و [D4-C1].',
  '',
  '- مورد نخست',
  '- مورد دوم',
  '',
  '1. گام اول',
  '2. گام دوم',
  '',
  '```',
  '<script>alert(1)</script>',
  '```',
  '',
  '<img src=x onerror=alert(1)>',
].join('\n'));

assert.equal(bubble.children[0].tagName, 'H1');
assert.ok(bubble.children.some(node => node.tagName === 'UL'));
assert.ok(bubble.children.some(node => node.tagName === 'OL'));
assert.ok(bubble.children.some(node => node.tagName === 'PRE'));
const paragraph = bubble.children.find(node => node.tagName === 'P');
assert.ok(paragraph.children.some(node => node.tagName === 'STRONG'));
assert.ok(paragraph.children.some(node => node.tagName === 'EM'));
assert.ok(paragraph.children.some(node => node.tagName === 'CODE'));
const citation = paragraph.children.find(node => node.className === 'citation');
assert.equal(citation.textContent, '[D4-C1]');
assert.equal(citation.dir, 'ltr');
assert.equal(bubble.textContent.includes('<img src=x onerror=alert(1)>'), true);
assert.equal(bubble.children.some(node => node.tagName === 'IMG'), false);

const parent = new Element('div');
parent.insertBefore = (node, reference) => {
  node.parentNode = parent;
  const index = parent.children.indexOf(reference);
  parent.children.splice(index < 0 ? parent.children.length : index, 0, node);
};
parent.append(bubble);
context.renderCitedSources(bubble, [
  {identifier: 'D4-C1', name: 'گزارش Project Alpha.pdf', location: 'صفحه ۲ تا ۳'},
  {identifier: 'D4-C1', name: 'نباید تکرار شود.pdf', location: ''},
  {identifier: 'D5-C2', name: 'بودجه.xlsx', location: 'کاربرگ بودجه، ردیف ۷ تا ۹'},
]);
const section = parent.children[1];
assert.equal(section.className, 'sources cited-sources');
const list = section.children[1];
assert.equal(list.childElementCount, 2);
assert.equal(section.textContent.includes('گزارش Project Alpha.pdf'), true);
assert.equal(section.textContent.includes('صفحه ۲ تا ۳'), true);
assert.equal(section.textContent.includes('نباید تکرار شود.pdf'), false);
