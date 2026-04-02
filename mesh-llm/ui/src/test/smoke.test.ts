/// <reference types="vitest/globals" />

import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('Vitest + testing library setup', () => {
  it('supports jsdom + jest-dom matchers', async () => {
    render(
      React.createElement(
        'button',
        { type: 'button', title: 'Smoke button' },
        'Smoke button'
      )
    );

    const button = screen.getByRole('button', { name: /smoke button/i });
    expect(button).toBeVisible();
    await userEvent.click(button);
    expect(button).toHaveTextContent('Smoke button');
    expect(button).toHaveAttribute('type', 'button');
  });
});
