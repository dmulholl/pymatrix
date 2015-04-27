"""
This plugin adds a string of page navigation links to index page objects.

The links can be accessed in templates via:

    {{ paging.links }}

Default settings can be overridden by including a 'paging' dictionary in
the site's config.py file containing one or more of the following options:

    paging = {
        'first': 'First',       # text for link to first page
        'last': 'Last',         # text for link to last page
        'prev': 'Prev',         # text for link to previous page
        'next': 'Next',         # text for link to next page
        'delta': 2,             # number of neighbouring pages to include
        'multiples': 2,         # number of larger/smaller multiples to include
        'multiple': 10,         # link to page numbers in multiples of...
    }

License: Public Domain.

"""

from ark import hooks, site


@hooks.register('rendering_page')
def add_paging_links(page):
    """ Adds a string of page navigation links to the Page object. """

    if page['paging']['is_paged']:
        page['paging']['links'] = generate_paging_links(
            page['slugs'][:-1],
            page['paging']['page'],
            page['paging']['total']
        )


def generate_paging_links(slugs, page_number, total_pages):
    """ Generates a string of page navigation links. """

    # Default settings can be overridden in the site's configuration file.
    data = {
        'first': 'First',
        'last': 'Last',
        'prev': 'Prev',
        'next': 'Next',
        'delta': 2,
        'multiples': 2,
        'multiple': 10,
    }
    data.update(site.config('paging', {}))

    # Start and end points for the sequence of numbered links.
    start = page_number - data['delta']
    end = page_number + data['delta']

    if start < 1:
        start = 1
        end = 1 + 2 * data['delta']

    if end > total_pages:
        start = total_pages - 2 * data['delta']
        end = total_pages

    if start < 1:
        start = 1

    out = []

    # First page link.
    if start > 1:
        out.append("<a class='first' href='%s'>%s</a>" % (
            site.paged_url(slugs, 1, total_pages),
            data['first']
        ))

    # Previous page link.
    if page_number > 1:
        out.append("<a class='prev' href='%s'>%s</a>" % (
            site.paged_url(slugs, page_number - 1, total_pages),
            data['prev']
        ))

    # Smaller multiple links.
    if data['multiples']:
        multiples = list(range(data['multiple'], start, data['multiple']))
        for multiple in multiples[-data['multiples']:]:
            out.append("<a class='pagenum multiple' href='%s'>%s</a>" % (
                site.paged_url(slugs, multiple, total_pages), multiple
            ))

    # Sequence of numbered page links.
    for i in range(start, end + 1):
        if i == page_number:
            out.append("<span class='pagenum current'>%s</span>" % i)
        else:
            out.append("<a class='pagenum' href='%s'>%s</a>" % (
                site.paged_url(slugs, i, total_pages), i
            ))

    # Larger multiple links.
    if data['multiples']:
        starting_multiple = (int(end / data['multiple']) + 1) * data['multiple']
        multiples = list(range(starting_multiple, total_pages, data['multiple']))
        for multiple in multiples[:data['multiples']]:
            out.append("<a class='pagenum multiple' href='%s'>%s</a>" % (
                site.paged_url(slugs, multiple, total_pages), multiple
            ))

    # Next page link.
    if page_number < total_pages:
        out.append("<a class='next' href='%s'>%s</a>" % (
            site.paged_url(slugs, page_number + 1, total_pages),
            data['next']
        ))

    # Last page link.
    if end < total_pages:
        out.append("<a class='last' href='%s'>%s</a>" % (
            site.paged_url(slugs, total_pages, total_pages),
            data['last']
        ))

    return ''.join(out)
